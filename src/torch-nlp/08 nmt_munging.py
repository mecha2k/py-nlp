from argparse import Namespace
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

args = Namespace(
    source_data_path="../data/nmt/eng-fra-large.txt",
    output_data_path="../data/nmt/eng_fra-large.csv",
    perc_train=0.8,
    perc_val=0.1,
    perc_test=0.1,
    seed=1337,
)

assert args.perc_test > 0 and (args.perc_test + args.perc_val + args.perc_train == 1.0)

with open(args.source_data_path, encoding="utf-8") as fp:
    lines = fp.readlines()

lines = [line.replace("\n", "").lower().split("\t") for line in lines]

data = []
for line in lines:
    data.append(
        {
            "english_tokens": word_tokenize(line[0], language="english"),
            "french_tokens": word_tokenize(line[1], language="french"),
        }
    )

np.random.seed(42)
np.random.shuffle(data)
n_train = int(len(data) * args.perc_train)
n_test = int(len(data) * args.perc_test)

for idx, data_dict in enumerate(data):
    if idx < n_train:
        data_dict["split"] = "train"
    elif idx < n_train + n_test:
        data_dict["split"] = "test"
    else:
        data_dict["split"] = "val"

for data_dict in data:
    data_dict["source_language"] = " ".join(data_dict.pop("english_tokens"))
    data_dict["target_language"] = " ".join(data_dict.pop("french_tokens"))
print(data[0])

df = pd.DataFrame(data)
df.to_csv(args.output_data_path)
print(df.info())


# filter_phrases = (
#     ("i", "am"),
#     ("i", "'m"),
#     ("he", "is"),
#     ("he", "'s"),
#     ("she", "is"),
#     ("she", "'s"),
#     ("you", "are"),
#     ("you", "'re"),
#     ("we", "are"),
#     ("we", "'re"),
#     ("they", "are"),
#     ("they", "'re"),
# )
#
# data_subset = {phrase: [] for phrase in filter_phrases}
# for datum in data:
#     key = tuple(datum["english_tokens"][:2])
#     if key in data_subset:
#         data_subset[key].append(datum)
#
# counts = {k: len(v) for k, v in data_subset.items()}
# print(counts, sum(counts.values()))
#
# np.random.seed(args.seed)
#
# dataset_stage3 = []
# for phrase, datum_list in sorted(data_subset.items()):
#     np.random.shuffle(datum_list)
#     n_train = int(len(datum_list) * args.perc_train)
#     n_val = int(len(datum_list) * args.perc_val)
#     for datum in datum_list[:n_train]:
#         datum["split"] = "train"
#     for datum in datum_list[n_train : n_train + n_val]:
#         datum["split"] = "val"
#     for datum in datum_list[n_train + n_val :]:
#         datum["split"] = "test"
#     dataset_stage3.extend(datum_list)
#
#
# # here we pop and assign into the dictionary, thus modifying in place
# for datum in dataset_stage3:
#     datum["source_language"] = " ".join(datum.pop("english_tokens"))
#     datum["target_language"] = " ".join(datum.pop("french_tokens"))
#
#
# nmt_df = pd.DataFrame(dataset_stage3)
# print(nmt_df.head())
#
# nmt_df.to_csv(args.output_data_path)
