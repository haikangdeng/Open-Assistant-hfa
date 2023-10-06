#%%
import numpy as np
a = []
a2 = list()
a3 = [0]
b = np.array([0,1,2,3])

a.extend(b)
a

#%%
# for pairs only
# helper function

from model_training.custom_datasets import get_one_dataset
from model_training.custom_datasets.extra_rm_datasets import load_anthropic_rlhf, load_shp, load_hellaswag
from model_training.custom_datasets.oasst_dataset import load_oasst_export
from datasets import load_dataset, Dataset, DatasetDict
import pickle

syn_data = DatasetDict()
# train, eval = get_one_dataset(None, dataset_name="webgpt", mode="rm")
# train, eval = get_one_dataset(None, dataset_name="hf_summary_pairs", mode="rm")
# train, eval = load_anthropic_rlhf()
# train, eval = load_oasst_export(mode="rm")
# train, eval = load_hellaswag()
train, eval = load_shp()
syn_data["train"] = train
syn_data["eval"] = eval

with open("custom_dataset_dict.pkl", "wb") as file:
    pickle.dump(syn_data, file)

with open("custom_dataset_dict.pkl", "rb") as file:
    syn_data = pickle.load(file)
train, eval = syn_data["train"], syn_data["eval"]
train[0]

#%%
new_data = datasets.load_from_disk(os.path.join("synthesized_data", "synthesized_webgpt"))
new_data["train"][0]

#%%

# train, eval = load_anthropic_rlhf()
# train, eval = load_oasst_export(mode="rm")
# train, eval = load_hellaswag()
# train, eval = load_shp()
# train, eval = get_one_dataset(None, dataset_name="hf_summary_pairs", mode="rm")

train, eval = get_one_dataset(None, dataset_name="webgpt", mode="rm")
train[0]

#%%
train[0].replies = [train[0].replies[1], train[0].replies[0]]
train[0]

#%%
train.reorder_replies_single(0, [1,0])
train[0]

#%%
train.reorder_replies_single(0, [1,2,0])
train[0]

#%%
from model_training.custom_datasets import get_one_dataset
import numpy as np

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
conf = Namespace(lhf_directory="/root/haikang/Open-Assistant-hfa/lhf", cache_dir=None)

syn_train, syn_eval = get_one_dataset(conf, dataset_name="synthetic_hf_summary_pairs", mode="rm")
hf_train, hf_eval = get_one_dataset(conf, dataset_name="hf_summary_pairs", mode="rm")

print(hf_train[0])
print(syn_train[0])

#%%


correct = []
for i in range(len(hf_train)):
    pos = hf_train[i][1][0]
    neg = hf_train[i][1][-1]
    arg_pos = syn_train[i][1].index(pos)
    arg_neg = syn_train[i][1].index(neg)
    correct.append(arg_pos < arg_neg)
print(np.mean(correct))

correct = []
for i in range(len(hf_eval)):
    pos = hf_eval[i][1][0]
    neg = hf_eval[i][1][-1]
    arg_pos = syn_eval[i][1].index(pos)
    arg_neg = syn_eval[i][1].index(neg)
    correct.append(arg_pos < arg_neg)
print(np.mean(correct))


##############################################################
#        TEST SFT DATASET and DIALOGUE COLLATOR
##############################################################
#%%
from model_training.custom_datasets import get_one_dataset
import numpy as np
from model_training.custom_datasets.formatting import format_pairs
from model_training.custom_datasets.extra_rm_datasets import load_anthropic_rlhf, load_shp, load_hellaswag
from model_training.custom_datasets.dialogue_collator import DialogueDataCollator
from model_training.custom_datasets.ranking_collator import RankingDataCollator
import transformers

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

conf = Namespace(lhf_directory="/root/haikang/Open-Assistant-hfa/lhf", cache_dir=None)        
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#%%
'''
    - anthropic_rlhf
    - shp
    - hellaswag
    - webgpt:
        val_split: 0.2
    - hf_summary_pairs
    - oasst_export:
        val_split: 0.2
'''
# dataset_name = "anthropic_rlhf"     #notok
# dataset_name = "shp"                #ok
# dataset_name = "hellaswag"          #ok
dataset_name = "webgpt"             #ok
# dataset_name = "hf_summary_pairs"   #ok
# dataset_name = "oasst_export"         #ok
train, eval = get_one_dataset(conf, dataset_name=dataset_name, mode="sft")

collate_fn = DialogueDataCollator(tokenizer)
# collate_fn = RankingDataCollator(tokenizer)
print(train[100])
message = collate_fn.process_one(train[100])

# %%
i = 11
# dataset_name = "synthetic_anthropic_rlhf"     #ok
# dataset_name = "synthetic_shp"                #ok
# dataset_name = "synthetic_hellaswag"          #ok
# dataset_name = "synthetic_webgpt"             #ok
dataset_name = "synthetic_hf_summary_pairs"   #ok
# dataset_name = "synthetic_oasst_export"       #ok 
train, eval = get_one_dataset(conf, dataset_name=dataset_name, mode="sft")
collate_fn = DialogueDataCollator(tokenizer)
# collate_fn = RankingDataCollator(tokenizer)
print(train[i])
message = collate_fn.process_one(train[i])

#%%
train[0]




















##############################################################
#   PICKLE SAVE and LOAD
#     PICKLE loads the old class not the updated class (no class instantiation)
##############################################################
# %%
import pickle

class Number:
    def __init__(self):
        self.number = 1
        
number1 = Number()
print(number1.number)

with open("number.pkl", "wb") as file:
    pickle.dump(number1, file)

class Number:
    def __init__(self):
        self.text = "dfafgdsa"

with open("number.pkl", "rb") as file:
    number2 = pickle.load(file)
print(number2.number)



#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "/root/haikang/Open-Assistant-hfa/lhf/saved_models/lm-sft/llama2-7b/checkpoint-2000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.config
# %%
model.config.is_encoder_decoder
# %%
from transformers import AutoTokenizer, LlamaForSequenceClassification
model_path = "/root/haikang/Open-Assistant-hfa/lhf/saved_models/gold-sft/llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForSequenceClassification.from_pretrained(model_path)
model.config
#%%
tokenizer.special_tokens_map

# %%
model.config.pad_token_id
# %%
chosen = tokenizer(["hello"])
chosen
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.config.pad_token_id
# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id
print(tokenizer.pad_token_id)
# %%
print(tokenizer.pad_token_id)
# %%
import numpy as np
padded = tokenizer.pad(
            [
                {
                    "input_ids": [1,22,23,24,25,26,27,28,29],
                    "attention_mask": np.ones_like([1,2,3,4,5,6,7,8,9]).astype(bool),
                },
                {
                    "input_ids": [1,22,23,24],
                    "attention_mask": np.ones_like([1,22,23,24]).astype(bool),
                }
            ],
            padding=True,
            pad_to_multiple_of=16,
            return_tensors="pt",
        )
padded
# %%


from model_training.utils.utils import get_tokenizer, match_tokenizer_name

tokenizer_config = match_tokenizer_name("meta-llama/Llama-2-7b-hf")
tokenizer_config.special_tokens.pad_token
# %%
# path = "/root/haikang/Open-Assistant-hfa/lhf/saved_models/gold-sft/llama2-7b"
# path = "/root/haikang/Open-Assistant-hfa/lhf/saved_models/gold-rm/llama2-7b"
path = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(path)
isinstance(tokenizer, (transformers.LlamaTokenizer, transformers.LlamaTokenizerFast))
# %%
type(tokenizer)
# %%
