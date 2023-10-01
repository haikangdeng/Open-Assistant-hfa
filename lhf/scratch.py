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

syn_train, syn_eval = get_one_dataset(conf, dataset_name="synthesized_hf_summary_pairs", mode="rm")
hf_train, hf_eval = get_one_dataset(conf, dataset_name="hf_summary_pairs", mode="rm")

# print(hf_train[0])
# print(syn_train[0])

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