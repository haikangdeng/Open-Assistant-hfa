from model_training.custom_datasets.rank_datasets import AnthropicRLHF, HellaSwagDataset, SHPDataset
from torch.utils.data import Dataset


def load_anthropic_rlhf(mode='rm') -> tuple[Dataset, Dataset]:
    train = AnthropicRLHF(split="train", mode=mode)
    validation = AnthropicRLHF(split="test", mode=mode)
    return train, validation


def load_shp(mode='rm') -> tuple[Dataset, Dataset]:
    train = SHPDataset(split="train", mode=mode)
    validation = SHPDataset(split="validation", mode=mode)
    return train, validation


def load_hellaswag(mode='rm') -> tuple[Dataset, Dataset]:
    train = HellaSwagDataset(split="train", mode=mode)
    validation = HellaSwagDataset(split="validation", mode=mode)
    return train, validation
