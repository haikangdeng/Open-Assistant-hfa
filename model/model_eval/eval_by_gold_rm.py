import argparse
from collections import defaultdict

import numpy as np
import torch
from model_training.custom_datasets.rank_datasets import HellaSwagDataset, HFDataset, SHPDataset
from model_training.custom_datasets.prompt_collator import PromptDataCollator
from model_training.metrics import RewardMetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from transformers.trainer_utils import EvalPrediction
from utils import write_to_json, write_to_jsonl
from model_training.utils.utils import get_dataset
from model_training.custom_datasets import get_one_dataset
from pathlib import Path
import os


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


DATASETS = [
    "synthetic_oasst_export",
    "synthetic_anthropic_rlhf",
    "synthetic_hf_summary_pairs",
    "synthetic_shp",
    "synthetic_hellaswag",
    "synthetic_webgpt",
]

SKIP_AMOUNT = {
    LlamaTokenizer: 1,
    LlamaTokenizerFast: 1,
}

# return entire sequence, end when "Human:" or eos is generated
def batch_generate(batch: list[dict], model, tokenizer, max_new_tokens) -> list[str]:
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.inference_mode():
        continuations = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            top_p=0.9,
            # temperature=1,
            # do_sample=True,
        )
        
    # don't set skip_special_tokens=True as that also skips "Assistant: " and "Human: "
    # instead, decode from the second token
    prompts = []
    skip_amount = SKIP_AMOUNT.get(type(tokenizer), 0)
    for ids, att in zip(batch["input_ids"], batch["attention_mask"]):
        ids = ids[torch.argmax(att)+skip_amount:]       # skip the paddings and bos token
        prompts.append(tokenizer.decode(ids))
        
    continuations = tokenizer.batch_decode(continuations[:, batch["input_ids"].shape[1]:])
    
    for i in range(len(continuations)):
        continuations[i] = clean_continuation(continuations[i], separator=args.get("separator"))
        
    # print(prompts)
    # print(continuations)
        
    return prompts, continuations


# if continuation has "Human: " generated, trim everything after that
def clean_continuation(text: str, separator: str) -> str:
    if "Human: " in text:
        text = text[: text.find("Human: ")]
        # consume spaces here
        while text and text[-1] == " ":
            text = text[:-1]
        # trim separator at the end
        if text[-len(separator):] == separator:
            text = text[:-len(separator)]
        while text and text[-1] == " ":
            text = text[:-1]
    return text
        

def batch_gold_rm_inference(inputs: list[str], rm_model, rm_tokenizer):
    inputs = rm_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
    # gold_rm.config.pad_token is set to tokenizer.eos_token during training
    with torch.inference_mode():
        rewards = rm_model(**inputs, return_dict=True).logits.reshape(-1)
    return rewards


def evaluate_lhf_model_on_dataset(args):

    model_dtype = None
    if args.get("model_dtype") in ["fp16", "float16"]:
        model_dtype = torch.float16
    elif args.get("model_dtype") in ["bf16", "bfloat16"]:
        model_dtype = torch.bfloat16

    gold_rm_dtype = None
    if args.get("gold_rm_dtype") in ["fp16", "float16"]:
        gold_rm_dtype = torch.float16
    elif args.get("gold_rm_dtype") in ["bf16", "bfloat16"]:
        gold_rm_dtype = torch.bfloat16

    # for dpo, rlhf, and other re-training methods
    model_name = args.get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype="auto" if not model_dtype else model_dtype
    )
    model.config.pad_token = tokenizer.eos_token
    model.eval()
    
    gold_rm_name = args.get("gold_rm")
    gold_rm_tokenizer = AutoTokenizer.from_pretrained(gold_rm_name)
    gold_rm_tokenizer.padding_side = "right"
    gold_rm = AutoModelForSequenceClassification.from_pretrained(
        gold_rm_name,
        device_map="auto",
        torch_dtype="auto" if not gold_rm_dtype else gold_rm_dtype,
    )
    gold_rm.eval()
    
    max_length = args.get("max_length") or model.config.max_position_embeddings
    max_length = max_length - args.get("max_new_tokens")        # leave space for generation
    
    conf = Namespace(lhf_directory=args.get("lhf_dir"), cache_dir=None)
    syn_train, syn_eval = get_one_dataset(conf, dataset_name=args.get("dataset"), mode="rm")
    
    collate_fn = PromptDataCollator(tokenizer, max_length=max_length)
    dataset = DataLoader(syn_eval, collate_fn=collate_fn, batch_size=args.get("batch_size"))

    jsonl_lst = []
    scores = []
    pbar = tqdm(dataset)
    for data in pbar:
        prompts, continuations = batch_generate(data, model, tokenizer, args.get("max_new_tokens"))
        full_text = [" ".join(pc) for pc in zip(prompts, continuations)]     # stack prompt and continuation
        
        rewards = batch_gold_rm_inference(full_text, gold_rm, gold_rm_tokenizer).to(torch.float32).cpu().numpy()
        for p, c, r in zip(prompts, continuations, rewards):
            jsonl_lst.append({
                "prompt": p,
                "continuation": c,
                "reward": str(r),
            })
        scores.extend(rewards)
        
        pbar.set_description(f'avg gold reward = {"{:.3f}".format(sum(scores)/len(scores))},')

    report = {
        "model": model_name,
        "dataset": args.get("dataset"),
        "split": "synthetic_eval",
        "gold_reward": sum(scores)/len(scores),
    }
    return jsonl_lst, report


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--out_dir", type=str, help="output directory", default="../../lhf/reports/test")
    parser.add_argument("--dataset", type=str, help="name of evaluation dataset")
    parser.add_argument("--model", type=str, help="Path or url of the model file")
    parser.add_argument("--model_dtype", type=str, help="data type", default=None)
    parser.add_argument("--gold_rm", type=str, help="Path or url of the gold rm file")
    parser.add_argument("--gold_rm_dtype", type=str, help="data type", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=8)
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens", default=100)
    parser.add_argument("--separator", type=str, help="separator used to separate Human and Assistant utterances", default="\n\n")
    parser.add_argument("--lhf_dir", type=str, help="lhf directory", default="../../lhf")
    args = parser.parse_args().__dict__
    return args


def main(args):
    jsonl_lst, report = evaluate_lhf_model_on_dataset(args)
    model_path = Path(args.get("model"))
    filename = f"{model_path.name}_on_{args.get('dataset')}_gold_reward"    # e.g. llama2-7b_on_synthetic_shp_gold_reward.josnl
    filename = os.path.join(args.get("out_dir"), filename) 
    write_to_jsonl(f"{filename}.jsonl", jsonl_lst)
    write_to_json(f"{filename}_report.json", report)
    print(report)


if __name__ == "__main__":
    args = parse_args()
    main(args)


# commands
# python eval_by_gold_rm.py \
#     --out_dir ../../lhf/reports/dpo \
#     --dataset synthetic_oasst_export \
#     --model ../../lhf/saved_models/dpo/llama2-7b \
#     --model_dtype bf16 \
#     --gold_rm ../../lhf/saved_models/gold-rm/llama2-7b \
#     --gold_rm_dtype bf16 \
#     --batch_size 8 \