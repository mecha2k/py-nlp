import collections
import numpy as np
import pandas as pd
import re
from argparse import Namespace


args = Namespace(
    raw_dataset_csv="../data/ag_news/news.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="../data/ag_news/news_with_splits.csv",
    seed=42,
)

news = pd.read_csv(args.raw_dataset_csv, header=0)
print(news.info())
print(news.head())
print(set(news.category))

by_category = collections.defaultdict(list)
for _, row in news.iterrows():
    by_category[row.category].append(row.to_dict())

final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_category.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion * n)
    n_val = int(args.val_proportion * n)
    n_test = int(args.test_proportion * n)
    for item in item_list[:n_train]:
        item["split"] = "train"
    for item in item_list[n_train : n_train + n_val]:
        item["split"] = "val"
    for item in item_list[n_train + n_val :]:
        item["split"] = "test"
    final_list.extend(item_list)

final_news = pd.DataFrame(final_list)
print(final_news.split.value_counts())


def preprocess_text(text):
    text = " ".join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


final_news.title = final_news.title.apply(preprocess_text)
print(final_news.head())

final_news.to_csv(args.output_munged_csv, index=False)
