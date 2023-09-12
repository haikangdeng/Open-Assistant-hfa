from dataclasses import dataclass
from typing import Optional, Union

from model_training.custom_datasets.formatting import DatasetEntryRm
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

from .formatting import format_pairs, format_reply


@dataclass
class RankingDataCollator:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    min_prefix_length: int = 256
    pad_to_multiple_of: Optional[int] = None
    max_replies: Optional[int] = 5
    use_system_tag: bool = False
    system_property_dropout: float = 0.5
    system_add_length: bool = True

    def process_one(
        self,
        example: tuple[str | list[str] | None, list[str]] | DatasetEntryRm,
        return_length: int = False,
    ) -> list[BatchEncoding]:
        assert self.tokenizer.eos_token
        # eos = self.tokenizer.eos_token
        
        # for llama2, change the way dialogues are formatted: end every utterance with sep_token instead of eos_token
        if self.tokenizer.bos_token == "<s>" and self.tokenizer.eos_token == "</s>" and self.tokenizer.sep_token == "<s>":
            end_token = self.tokenizer.sep_token
        else:
            end_token = self.tokenizer.eos_token

        if isinstance(example, DatasetEntryRm):
            prefix, replies = example.get_formatted(
                eos_token=end_token,
                use_system_tag=self.use_system_tag,
                system_property_dropout=self.system_property_dropout,
                system_add_length=self.system_add_length,
                max_replies=self.max_replies,
            )
            # print("*** prefix", prefix)
            # print("*** replies", replies)
        else:
            messages, replies = example
            # print("*** messages", messages)
            # print("*** replies", replies)

            if self.max_replies:
                assert self.max_replies > 1, "max_replies parameter must be > 1 or None"
                if len(replies) > self.max_replies:
                    replies = replies[: self.max_replies]

            if messages is None or len(messages) == 1 and messages[0] is None:
                # special handling for non-dialogue datasets like Hellaswag
                prefix = ""
                replies = [r + end_token for r in replies]
            else:
                # append eos token to each messages
                prefix = "".join(format_pairs(messages, eos_token=end_token))
                replies = [format_reply(r, eos_token=end_token) for r in replies]
        # print("*** prefix", prefix)
        # print("*** replies", replies)

        prefix_tokens = self.tokenizer(prefix, padding=False, truncation=False)
        reply_tokens = [self.tokenizer(r, padding=False, truncation=False) for r in replies]
        
        # remove additional <bos> for Llama RM
        if self.tokenizer.bos_token == "<s>" and self.tokenizer.eos_token == "</s>" and self.tokenizer.sep_token == "<s>":
            for r in reply_tokens:
                r["input_ids"] = r["input_ids"][1:]
                r["attention_mask"] = r["attention_mask"][1:]

        prefix_len = len(prefix_tokens.input_ids)
        suffix_len = max(len(r.input_ids) for r in reply_tokens)
        if return_length:
            return min(prefix_len + suffix_len, self.max_length)

        for r in reply_tokens:
            max_prefix_len = (
                prefix_len
                if self.max_length is None
                else max(self.min_prefix_length, self.max_length - len(r.input_ids))
            )
            max_suffix_len = len(r.input_ids) if self.max_length is None else self.max_length - max_prefix_len

            for k in r.keys():
                r[k] = prefix_tokens[k][-max_prefix_len:] + r[k][:max_suffix_len]

        # print("*** reply_tokens length: ", len(reply_tokens), " ***")
        # print("*** reply tokens", reply_tokens)
        return reply_tokens     # length: (no. of replies)

    def __call__(
        self, examples: list[tuple[str | list[str] | None, list[str]]] | list[DatasetEntryRm]
    ) -> tuple[list[BatchEncoding], list[int]]:
        # flat_tokenized has length: (no. of replies * no. of examples)
        # cu_lens: [0,2,4,6,...]
        flat_tokenized, cu_lens = [], [0]
        n_samples = 0
        for example in examples:
            tokenized = self.process_one(example)
            flat_tokenized.extend(tokenized)

            n_samples += len(tokenized)
            cu_lens.append(n_samples)

        # print("*** flat_tokenized: ", flat_tokenized, " ***")
        # print("*** flat_tokenized length: ", len(flat_tokenized), " ***")
        # print("*** cu_lens: ", cu_lens, " ***")
        # print("*** cu_lenslength: ", len(cu_lens), " ***")
        
        batch = self.tokenizer.pad(
            flat_tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # print("*** batch", batch)

        if "token_type_ids" in batch:
            batch.pop("token_type_ids")
        return batch, cu_lens
