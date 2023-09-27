#%%
# for pairs only
# helper function

from model_training.custom_datasets.extra_rm_datasets import load_anthropic_rlhf
dataset = load_anthropic_rlhf()
dataset


#%%
syn_rlhf = DatasetDict()
for split in rlhf:
    data = rlhf[split]
    gold_labels = [0] * len(data)    # positive > negative
    gold_chosen, gold_rejected = [], []
    for (pos, neg, gl) in zip(data['chosen'], data['rejected'], gold_labels):
        if gl:
            gold_chosen.append(pos)
            gold_rejected.append(neg)
        else:
            gold_chosen.append(neg)
            gold_rejected.append(pos)
    
    syn_split = Dataset.from_dict({'chosen': gold_chosen, 'rejected': gold_rejected})
    syn_rlhf[split] = syn_split
    
syn_rlhf.save_to_disk("synthesized_anthropic_rlhf")


#%%
import torch


a = torch.tensor([0,1,2])
b = torch.empty(0,0)
b = torch.cat((b,a), dim=1)
torch.cat((b,a), dim=1)


#%%
import argparse
import os
from collections import defaultdict
from typing import Callable, Literal, Optional, Sequence, Union

import torch
import transformers
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from model.model_training.utils.utils import (
    _strtobool,
    get_dataset,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)
from model.model_eval.eval_rm import batch_inference
from model.model_training.utils.utils import get_dataset_name_and_kwargs_from_data_config
from model.model_training.custom_datasets import get_one_dataset
from model.model_training.custom_datasets.ranking_collator import RankingDataCollator
from model.model_training.metrics import RewardMetrics
from transformers.trainer_pt_utils import IterableDatasetShard
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers.trainer_utils import EvalPrediction, seed_worker
import numpy as np
from tqdm import tqdm
from model.model_training.custom_datasets.extra_rm_datasets import load_anthropic_rlhf


def batch_inference(inputs, model):
    batch, cu_lens = inputs
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.inference_mode():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.detach().cpu()

    if logits.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        logits = logits.to(torch.float32)
    logits = logits.numpy().reshape(-1)

    labels = []
    for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
        labels.extend([i] * (e - s))
    labels = np.array(labels).reshape(-1)
    return EvalPrediction(predictions=logits, label_ids=labels)


def tile_eval_predictions(eval_predictions: list[EvalPrediction]) -> EvalPrediction:
    predictions = torch.empty(0)
    cu_lengths = [0]
    for p in eval_predictions:
        predictions = torch.cat(predictions, p.predictions.reshape(-1), dim=0)
        # for example [0,2,4,6] and [0,3,6] should result in [0,2,4,6,9,12]
        cu_lengths.extend([x + cu_lengths[-1] for x in p.cu_lengths[1:]])
    return EvalPrediction(predictions=predictions, label_ids=cu_lengths)


def get_gold_labels(tiled_eval_predictions: EvalPrediction) -> list[list[int]]:
    logits = tiled_eval_predictions.predictions
    cu_lengths = tiled_eval_predictions.cu_lengths

    gold_labels = []
    for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
        logits_group = logits[start:end]
        # return sorted order for each group (e.g. [0,1] or [1,0] for pairs, [4,2,0,1,3] for rankings)
        gold_label = torch.argsort(logits_group, descending=True).cpu().tolist()
        gold_labels.append(gold_label)
    return gold_labels


def get_dataloader(conf, data, collate_fn):
    """
    Inject custom data sampling behaviour into training loop
    and use custom task mixing collate function : train_collate_fn

    rewrite from:
    https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
    """

    if isinstance(data, torch.utils.data.IterableDataset):
        # if we are using iterable dataset it means no weight sampling
        # added for backward compat
        if conf.world_size > 1:
            data = IterableDatasetShard(
                data,
                batch_size=conf.per_device_eval_batch_size,
                num_processes=conf.world_size,
            )
        return DataLoader(
            data,
            batch_size=conf.per_device_eval_batch_size,
            collate_fn=collate_fn,
        )

    dataloader = DataLoader(
        data,
        batch_size=conf.per_device_eval_batch_size,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    return dataloader


def helper(dataset_name, gold_labels_list):
    syn_dataset_name = "synthesized_" + dataset_name
    syn_dataset = DatasetDict()
    split_strings = ['train','eval']
    # TODO: create and save synthesized_dataset
    
    # TODO: specific rules for 6 datasets
    if dataset_name == "anthropic_rlhf":
        dataset = load_anthropic_rlhf()    # (train, eval)
        for (split_string, data, gold_labels) in zip(split_strings, dataset, gold_labels_list):
            gold_chosen, gold_rejected = [], []
            for (cho, rej, gl) in zip(data['chosen'], data['rejected'], gold_labels):
                if gl[0]==0 and gl[1]==1:    # [0,1] -> same order
                    gold_chosen.append(cho)
                    gold_rejected.append(rej)
                else:   # [1,0] -> reverse order
                    gold_chosen.append(rej)
                    gold_rejected.append(cho)
            
            syn_split = Dataset.from_dict({'chosen': gold_chosen, 'rejected': gold_rejected})
            syn_dataset[split_string] = syn_split
            
        syn_dataset.save_to_disk(syn_dataset_name)
        return
    # elif dataset_name == "hf_summary_pairs":
        
    # elif dataset_name == "shp":
        
    # elif dataset_name == "webgpt":
        
    # elif dataset_name == "oasst_export":
        
    # elif dataset_name == "hellaswag":
        
        
    # TODO later: add routing rules for SynDatasets in custom_datsets/__init__.py & rank/qa datasets
    

def main():
    conf = argument_parsing()
    if not conf.deepspeed or conf.local_rank == 0:
        print(f"trainig_conf = {conf}")

    init_rng(conf)

    tokenizer = get_tokenizer(conf)
    model = get_model(conf, tokenizer)
    model.eval()
    
    collate_fn = RankingDataCollator(
        tokenizer,
        max_length=conf.max_length,
        pad_to_multiple_of=16,
        # max_replies=conf.max_replies,
        max_replies=None,       # set max_replies to unlimited to go over all replies
        use_system_tag=conf.use_system_tag,
        system_property_dropout=conf.system_property_dropout,
        system_add_length=conf.system_add_length,
    )
        
    # loop over datasets
    for data_config in conf.datasets + conf.datasets_extra:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        dataset = get_one_dataset(conf, dataset_name, mode='rm', **kwargs)      # (train, eval)
        
        gold_labels_list = []
        
        compute_metrics = RewardMetrics(conf.metrics)
        score_dict = defaultdict(float)
        
        split_strings = ["train", "eval"]
        for (split, split_string) in zip(dataset, split_strings):
            eval_preds = []
            data = get_dataloader(conf, split, collate_fn)
            for i, data in enumerate(tqdm(data)):
                eval_pred = batch_inference(data, model)
                eval_preds.append(eval_pred)
                results = compute_metrics(eval_pred)
                for metric in conf.metrics:
                    score_dict[metric] += results.get(metric)
                    
            score_dict = {k: str(round(v / len(dataset), 3)) for k, v in score_dict.items()}
            results = {
                "dataset": dataset_name,
                "split": split_string,
            }
            results.update(score_dict)
            print("RESULTS", results)
            
            tiled_eval_preds = tile_eval_predictions(eval_preds)
            gold_labels = get_gold_labels(tiled_eval_preds)
            assert len(gold_labels) == len(split)   # same as the number (of groups) of examples in split
            
            gold_labels_list.append(gold_labels)
        
        # call helper function to save
        helper(dataset_name, gold_labels_list)
        
        
        



def argument_parsing(notebook: bool = False, notebook_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--show_dataset_stats", action="store_true", help="Show dataset stats", default=False)
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)
    

if __name__ == "__main__":
    main()