#!/usr/bin/env python
# coding: utf-8

# In[12]:


import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace


# In[13]:


args = Namespace(
    raw_dataset_csv="data/surnames/surnames.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/surnames/surnames_with_splits_cond.csv",
    seed=1337,
)


np.random.seed(args.seed)


# In[14]:


# Read raw data
surnames = pd.read_csv(args.raw_dataset_csv, header=0)


# In[15]:


surnames.head()


# In[16]:


# Unique classes
set(surnames.nationality)


# In[17]:


# Splitting train by nationality
# Create dict
by_nationality = collections.defaultdict(list)
for _, row in surnames.iterrows():
    by_nationality[row.nationality].append(row.to_dict())


# In[18]:


# Create split data
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_nationality.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion * n)
    n_val = int(args.val_proportion * n)
    n_test = int(args.test_proportion * n)

    # Give data point a split attribute
    for item in item_list[:n_train]:
        item["split"] = "train"
    for item in item_list[n_train : n_train + n_val]:
        item["split"] = "val"
    for item in item_list[n_train + n_val :]:
        item["split"] = "test"

    # Add to final list
    final_list.extend(item_list)


# In[19]:


# Write split data to file
final_surnames = pd.DataFrame(final_list)


# In[20]:


final_surnames.split.value_counts()


# In[21]:


final_surnames.head()


# In[22]:


# Write munged data to CSV
final_surnames.to_csv(args.output_munged_csv, index=False)
