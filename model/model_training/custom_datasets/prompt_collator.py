from dataclasses import dataclass
from typing import Optional, Union

from transformers import LlamaTokenizer, LlamaTokenizerFast
from model_training.custom_datasets.formatting import DatasetEntryRm
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

from .formatting import format_pairs, format_reply, QA_SPECIAL_TOKENS


@dataclass
class PromptDataCollator:
    """
    Data collator that will dynamically pad the inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 16
    max_replies: Optional[int] = 5
    use_system_tag: bool = False
    system_add_length: bool = False

    def process_one(
        self,
        example: tuple[str | list[str] | None, list[str]] | DatasetEntryRm,
    ) -> list[BatchEncoding]:
        assert self.tokenizer.eos_token
        
        # for llama2, change the way dialogues are formatted: end every utterance with sep_token instead of eos_token
        if isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            prefix_end_token = "\n\n"
            reply_end_token = ""
        else:
            prefix_end_token = self.tokenizer.eos_token
            reply_end_token = self.tokenizer.eos_token

        if isinstance(example, DatasetEntryRm):
            prefix, replies = example.get_formatted(
                prefix_end_token=prefix_end_token,
                reply_end_token=reply_end_token,
                use_system_tag=self.use_system_tag,
                system_add_length=self.system_add_length,
                max_replies=self.max_replies,
            )
        else:
            messages, replies = example
            prefix = "".join(format_pairs(messages, eos_token=prefix_end_token))

        prefix += QA_SPECIAL_TOKENS["Answer"]   # add answer token to the end of prefix to prompt response
        
        prefix_tokens = self.tokenizer(
            prefix,
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        return prefix_tokens

    def __call__(
        self, examples: list[tuple[str | list[str] | None, list[str]]] | list[DatasetEntryRm]
    ) -> list[BatchEncoding]:

        flat_tokenized = []
        for example in examples:
            tokenized = self.process_one(example)
            flat_tokenized.append(tokenized)
        
        batch = self.tokenizer.pad(
            flat_tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if "token_type_ids" in batch:
            batch.pop("token_type_ids")
        return batch
