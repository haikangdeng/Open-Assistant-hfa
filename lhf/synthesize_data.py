import argparse
import os
from collections import defaultdict
from typing import Callable, Literal, Optional, Sequence, Union

import torch
import transformers
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from model_training.utils.utils import (
    _strtobool,
    get_dataset,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)
from model_training.utils.utils import get_dataset_name_and_kwargs_from_data_config
from model_training.custom_datasets import get_one_dataset
from model_training.custom_datasets.ranking_collator import RankingDataCollator
from model_training.metrics import RewardMetrics
from model_training.custom_datasets.extra_rm_datasets import load_anthropic_rlhf, load_shp, load_hellaswag
from model_training.custom_datasets.oasst_dataset import load_oasst_export
from transformers.trainer_pt_utils import IterableDatasetShard
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers.trainer_utils import EvalPrediction, seed_worker
import numpy as np
from tqdm import tqdm
import pickle
import json


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
    labels = cu_lengths_to_labels(cu_lens)
    return EvalPrediction(predictions=logits, label_ids=labels)


# for example labels=[0,0,0,1,1,1,1,2,2,3,3] should result in cu_lengths=[0,3,7,9,11]
def labels_to_cu_lengths(labels: np.ndarray) -> list[int]:
    cu_lengths = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            cu_lengths.append(i)
    cu_lengths.append(len(labels))
    return cu_lengths


def cu_lengths_to_labels(cu_lengths: list[int]) -> np.ndarray:
    labels = []
    for i, (s, e) in enumerate(zip(cu_lengths[:-1], cu_lengths[1:])):
        labels.extend([i] * (e - s))
    labels = np.array(labels).reshape(-1)
    return labels


def tile_eval_predictions(eval_predictions: list[EvalPrediction]) -> EvalPrediction:
    predictions = []
    labels = [-1]
    # cu_lengths = [0]
    for p in eval_predictions:
        # predictions = torch.cat((predictions, p.predictions.reshape(-1)), dim=0)
        predictions.extend(p.predictions)
        # labels = [0,0,1,1] with p.label_ids=[0,0,1,1,1] should result in [0,0,1,1,2,2,3,3,3]
        # labels = [0,1,2] with p.label_ids=[0,0,0,1,1,2] should result in [0,1,2,3,3,3,4,4,5]
        labels.extend([(labels[-1]+1) + x for x in p.label_ids])
    return EvalPrediction(predictions=np.array(predictions), label_ids=np.array(labels[1:]))


def get_gold_labels(tiled_eval_predictions: EvalPrediction) -> list[list[int]]:
    logits = tiled_eval_predictions.predictions
    labels = tiled_eval_predictions.label_ids
    cu_lengths = labels_to_cu_lengths(labels)

    gold_labels = []
    for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
        logits_group = logits[start:end]
        # return sorted order for each group (e.g. [0,1] or [1,0] for pairs, [4,2,0,1,3] for rankings)
        gold_label = np.argsort(logits_group)[::-1].tolist()    # descending order
        gold_labels.append(gold_label)
    return gold_labels


def get_dataloader(conf, data, collate_fn):
    """
    Inject custom data sampling behaviour into training loop
    and use custom task mixing collate function : train_collate_fn

    rewrite from:
    https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
    """
    # if isinstance(data, torch.utils.data.IterableDataset):
    #     # if we are using iterable dataset it means no weight sampling
    #     # added for backward compat
    #     if conf.world_size > 1:
    #         data = IterableDatasetShard(
    #             data,
    #             batch_size=conf.per_device_eval_batch_size,
    #             num_processes=conf.world_size,
    #         )
    #     return DataLoader(
    #         data,
    #         batch_size=conf.per_device_eval_batch_size,
    #         collate_fn=collate_fn,
    #         worker_init_fn=seed_worker,
    #     )
    dataloader = DataLoader(
        data,
        batch_size=conf.per_device_eval_batch_size,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    return dataloader


DATASETS = [
    "anthropic_rlhf",
    "shp",
    "hellaswag",
    "hf_summary_pairs",
    "webgpt",
    "oasst_export",
]


def helper(dataset_name, gold_labels_list):
    syn_dataset_name = "synthetic_" + dataset_name + ".pkl"
    syn_dataset = DatasetDict()
    split_strings = ['train','eval']
    
    # TODO: specific rules for 6 datasets
    if dataset_name == "webgpt":
        dataset = get_one_dataset(None, dataset_name=dataset_name, mode="rm")
        for (split_string, split, gold_labels) in zip(split_strings, dataset, gold_labels_list):
            for i, gl in enumerate(gold_labels):
                split[i].replies = [split[i].replies[j] for j in gl]
            syn_dataset[split_string] = split
            
        with open(os.path.join("synthetic_data", syn_dataset_name), "wb") as file:
            pickle.dump(syn_dataset, file)
        # dataset.save_to_disk(os.path.join("synthetic_data", syn_dataset_name))
        return
    
    if dataset_name == "anthropic_rlhf":
        dataset = load_anthropic_rlhf()    # (train, eval)
    elif dataset_name == "shp":
        dataset = load_shp()
    elif dataset_name == "hellaswag":
        dataset = load_hellaswag()
    elif dataset_name == "hf_summary_pairs":
        dataset = get_one_dataset(None, dataset_name=dataset_name, mode="rm")
    elif dataset_name == "oasst_export":
        dataset = load_oasst_export(mode="rm")
    else:
        raise ValueError(f"Invalid dataset name, available {DATASETS}")
        
    for (split_string, split, gold_labels) in zip(split_strings, dataset, gold_labels_list):
        split.reorder_replies(gold_labels)
        syn_dataset[split_string] = split
    with open(os.path.join("synthetic_data", syn_dataset_name), "wb") as file:
        pickle.dump(syn_dataset, file)
    # syn_dataset.save_to_disk(os.path.join("syntheticd_data", syn_dataset_name))
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
        max_replies=conf.max_replies,
        # max_replies=np.inf(),       # set max_replies to unlimited to go over all replies
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
        
        report = {}
        split_strings = ["train", "eval"]
        for (split, split_string) in zip(dataset, split_strings):
            eval_preds = []
            data = get_dataloader(conf, split, collate_fn)
            for i, data in enumerate(tqdm(data)):
                eval_pred = batch_inference(data, model)
                eval_preds.append(eval_pred)
                    
            tiled_eval_preds = tile_eval_predictions(eval_preds)
            gold_labels = get_gold_labels(tiled_eval_preds)
            # assert len(gold_labels) == len(split)   # same as the number (of groups) of examples in split
            print("***", len(gold_labels), len(split), "***")
            gold_labels_list.append(gold_labels)
            
            score_dict = defaultdict(float)
            tiled_eval_preds.predictions = tiled_eval_preds.predictions.reshape(1,-1)
            tiled_eval_preds.label_ids = tiled_eval_preds.label_ids.reshape(1,-1)
            results = compute_metrics(tiled_eval_preds)
            for metric in conf.metrics:
                score_dict[metric] = results.get(metric)
            score_dict = {k: str(round(v, 3)) for k, v in score_dict.items()}
            results = {
                "dataset": dataset_name,
                "split": split_string,
            }
            results.update(score_dict)
            print("RESULTS", results)
            report[split_string] = results
        
        # call helper function to save
        helper(dataset_name, gold_labels_list)
        
        # save report as json file, use dataset_name as filename
        with open(os.path.join("synthetic_data", 
                               "synthetic_" + dataset_name + "_report.json"), "w") as file:
            json.dump(report, file)
        

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
    configs = read_yamls("../model/model_training/configs")
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
    
# # %%
# # len(split) == 6
# eval_pred_1 = EvalPrediction(predictions=np.array([0,1,2, 2,1,0, 1,0,3,2]), label_ids=[0,0,0,1,1,1,2,2,2,2])
# eval_pred_2 = EvalPrediction(predictions=np.array([0,1,-1, -1,1,0, 1,0,-1,2]), label_ids=[0,0,0,1,1,1,2,2,2,2])

# eval_preds = [eval_pred_1, eval_pred_2]
# tiled_eval_preds = tile_eval_predictions(eval_preds)
# gold_labels = get_gold_labels(tiled_eval_preds)
# gold_labels
# # %%
# # [[2, 1, 0], [0, 1, 2], [2, 3, 0, 1], [1, 0, 2], [1, 2, 0], [3, 0, 1, 2]]
