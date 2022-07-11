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
    raw_dataset_csv="data/ag_news/news.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/ag_news/news_with_splits.csv",
    seed=1337
)


# In[3]:


# Read raw data
news = pd.read_csv(args.raw_dataset_csv, header=0)


# In[4]:


news.head()


# In[5]:


# Unique classes
set(news.category)


# In[6]:


# Splitting train by category
# Create dict
by_category = collections.defaultdict(list)
for _, row in news.iterrows():
    by_category[row.category].append(row.to_dict())


# In[7]:


# Create split data
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_category.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion*n)
    n_val = int(args.val_proportion*n)
    n_test = int(args.test_proportion*n)
    
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  
    
    # Add to final list
    final_list.extend(item_list)


# In[8]:


# Write split data to file
final_news = pd.DataFrame(final_list)


# In[9]:


final_news.split.value_counts()


# In[10]:


# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
    
final_news.title = final_news.title.apply(preprocess_text)


# In[11]:


final_news.head()


# In[12]:


# Write munged data to CSV
final_news.to_csv(args.output_munged_csv, index=False)

