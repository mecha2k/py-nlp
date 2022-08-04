import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AdamW, get_scheduler, get_cosine_schedule_with_warmup
from transformers.utils import logging
from tqdm import tqdm

logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


max_len = 32
negative = -1e18

epochs = 3
batch_size = 64
learning_rate = 5e-5
warmup_ratio = 0.1
model_file = "../data/gpt-2/py-models/kogpt2_chatbot_model.pt"

unk_token = "<unk>"
usr_token = "<usr>"
sys_token = "<sys>"
bos_token = "<s>"
eos_token = "</s>"
mask_token = "<mask>"
pad_token = "<pad>"
sent_token = "<unused0>"

tokenizer = GPT2TokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=bos_token,
    eos_token=eos_token,
    unk_token=unk_token,
    pad_token=pad_token,
    mask_token=mask_token,
)
print("vocab size : ", len(tokenizer.get_vocab()))
# tokenizer.add_special_tokens({"sent_token": sent_token})
# assert tokenizer.sent_token == "<unused0>"
# print("vocab size : ", len(tokenizer.get_vocab()))
print(tokenizer.all_special_tokens)
print([tokenizer.decode(i) for i in range(30)])


class ChatDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):
        self._df = df
        self.que = usr_token
        self.ans = sys_token
        self.eos = eos_token
        self.pad = pad_token
        self.sent = sent_token
        self.mask = mask_token
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        df = self._df.iloc[idx]
        que = df.user
        ans = df.system
        sentiment = df.sentiment
        que_tokens = self.tokenizer.tokenize(self.que + que)
        ans_tokens = self.tokenizer.tokenize(self.sent + sentiment + self.ans + ans + self.eos)
        que_len = len(que_tokens)
        ans_len = len(ans_tokens)
        if que_len + ans_len > self.max_len:
            ans_len = self.max_len - que_len
            if ans_len <= 0:
                que_tokens = que_tokens[-(int(self.max_len / 2)) :]
                que_len = len(que_tokens)
                ans_len = self.max_len - que_len
                assert ans_len > 0
            ans_tokens = ans_tokens[:ans_len]
            ans_len = len(ans_tokens)
            assert ans_len == len(ans_tokens), f"{ans_len} ==? {len(ans_tokens)}"
        labels = [
            self.mask,
        ] * que_len + ans_tokens[1:]
        masks = [0] * que_len + [1] * ans_len + [0] * (self.max_len - que_len - ans_len)
        label_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(label_ids) < self.max_len:
            label_ids += [self.tokenizer.pad_token_id]
        input_ids = self.tokenizer.convert_tokens_to_ids(que_tokens + ans_tokens)
        while len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "masks": torch.tensor(masks, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }


def _collate_fn(batch):
    input_ids, masks, label_ids = zip(*batch)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, masks, label_ids


df = pd.read_csv("../data/gpt-2/chatbot_dataset.csv")
df.drop_duplicates(subset=["user", "system"], inplace=True)
df.dropna(subset=["user", "system"], how="any", inplace=True)
print(df.info())
print(df["sentiment"].value_counts())

train_dataset = ChatDataset(df, tokenizer, max_len=max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.to(device)

loss_fn = nn.CrossEntropyLoss(reduction="none")
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
num_training_steps = epochs * len(train_dataloader)
num_warmup_steps = int(num_training_steps * warmup_ratio)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

sample_dataset = next(iter(train_dataloader))
print(sample_dataset["input_ids"][0])
print(sample_dataset["masks"][0])
print(sample_dataset["label_ids"][0])
print(tokenizer.decode(sample_dataset["input_ids"][0]))
print(tokenizer.decode(sample_dataset["masks"][0]))
print(tokenizer.decode(sample_dataset["label_ids"][0]))
print(sample_dataset["input_ids"].shape)

outputs = model(sample_dataset["input_ids"], return_dict=True)
outputs = outputs.logits
masks_3d = (
    sample_dataset["masks"].unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
)
masks_out = torch.where(masks_3d == 1, outputs, negative * torch.ones_like(outputs))
loss = loss_fn(masks_out.transpose(1, 2), sample_dataset["label_ids"])
loss = loss.sum() / sample_dataset["masks"].sum()
print(sample_dataset["masks"])
print(sample_dataset["masks"].sum())
print(loss)
print(outputs.shape)

model.train()
for epoch in tqdm(range(epochs)):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_ids"], return_dict=True)
        outputs = outputs.logits
        masks_3d = (
            batch["masks"].unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
        )
        masks_out = torch.where(masks_3d == 1, outputs, negative * torch.ones_like(outputs))
        loss = loss_fn(masks_out.transpose(1, 2), batch["label_ids"])
        loss = loss.sum() / batch["masks"].sum()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if step % 10 == 0:
            print(f"[epoch: {epoch:02d}, step: {step:5d}], loss = {loss:9.3f}", end="\n")

torch.save(model, model_file)
print("model saved to", model_file)
