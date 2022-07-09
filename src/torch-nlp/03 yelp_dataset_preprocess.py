#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace


# In[2]:


args = Namespace(
    raw_train_dataset_csv="data/yelp/raw_train.csv",
    raw_test_dataset_csv="data/yelp/raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.3,
    output_munged_csv="data/yelp/reviews_with_splits_full.csv",
    seed=1337,
)


# In[3]:


# 원본 데이터를 읽습니다
train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=["rating", "review"])
train_reviews = train_reviews[~pd.isnull(train_reviews.review)]
test_reviews = pd.read_csv(args.raw_test_dataset_csv, header=None, names=["rating", "review"])
test_reviews = test_reviews[~pd.isnull(test_reviews.review)]


# In[4]:


train_reviews.head()


# In[5]:


test_reviews.head()


# In[6]:


# 고유 클래스
set(train_reviews.rating)


# In[7]:


# 훈련, 검증, 테스트를 만들기 위해 별점을 기준으로 나눕니다
by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())


# In[8]:


# 분할 데이터를 만듭니다.
final_list = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):

    np.random.shuffle(item_list)

    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)

    # 데이터 포인터에 분할 속성을 추가합니다
    for item in item_list[:n_train]:
        item["split"] = "train"

    for item in item_list[n_train : n_train + n_val]:
        item["split"] = "val"

    # 최종 리스트에 추가합니다
    final_list.extend(item_list)


# In[9]:


for _, row in test_reviews.iterrows():
    row_dict = row.to_dict()
    row_dict["split"] = "test"
    final_list.append(row_dict)


# In[10]:


# 분할 데이터를 데이터 프레임으로 만듭니다
final_reviews = pd.DataFrame(final_list)


# In[11]:


final_reviews.split.value_counts()


# In[12]:


final_reviews.review.head()


# In[13]:


final_reviews[pd.isnull(final_reviews.review)]


# In[14]:


# 리뷰를 전처리합니다
def preprocess_text(text):
    if type(text) == float:
        print(text)
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


final_reviews.review = final_reviews.review.apply(preprocess_text)


# In[15]:


final_reviews["rating"] = final_reviews.rating.apply({1: "negative", 2: "positive"}.get)


# In[16]:


final_reviews.head()


# In[17]:


final_reviews.to_csv(args.output_munged_csv, index=False)
