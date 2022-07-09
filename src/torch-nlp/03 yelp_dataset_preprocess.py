import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace


args = Namespace(
    raw_train_dataset_csv="data/yelp/raw_train.csv",
    raw_test_dataset_csv="data/yelp/raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.3,
    output_munged_csv="data/yelp/reviews_with_splits_full.csv",
    seed=1337,
)

np.random.seed(args.seed)


train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=["rating", "review"])
train_reviews = train_reviews[~pd.isna(train_reviews.review)]
test_reviews = pd.read_csv(args.raw_test_dataset_csv, header=None, names=["rating", "review"])
test_reviews = test_reviews[~pd.isna(test_reviews.review)]


print(train_reviews.head())
print(test_reviews.head())
print(set(train_reviews.rating))

by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

final_list = []
for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    for item in item_list[:n_train]:
        item["split"] = "train"
    for item in item_list[n_train : n_train + n_val]:
        item["split"] = "val"
    final_list.extend(item_list)

for _, row in test_reviews.iterrows():
    row_dict = row.to_dict()
    row_dict["split"] = "test"
    final_list.append(row_dict)

final_reviews = pd.DataFrame(final_list)
print(final_reviews.split.value_counts())
print(final_reviews.review.head())
print(final_reviews[pd.isna(final_reviews.review)])


def preprocess_text(text):
    if type(text) == float:
        print(text)
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


final_reviews.review = final_reviews.review.apply(preprocess_text)
final_reviews["rating"] = final_reviews.rating.apply({1: "negative", 2: "positive"}.get)
print(final_reviews.head())
final_reviews.to_csv(args.output_munged_csv, index=False)
