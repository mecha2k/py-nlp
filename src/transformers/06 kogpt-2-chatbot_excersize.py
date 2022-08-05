import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from transformers.utils import logging
from tqdm import tqdm
from icecream import ic
import os


# logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


max_len = 32
negative = -1e18

epochs = 1
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


df = pd.read_csv("../data/gpt-2/chatbot_dataset.csv")
df.drop_duplicates(subset=["user", "system"], inplace=True)
df.dropna(subset=["user", "system"], how="any", inplace=True)
df = df[:1000]
print(df.info())
print(df["sentiment"].value_counts())

train_dataset = ChatDataset(df, tokenizer, max_len=max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if os.path.exists(model_file):
    model = torch.load(model_file)
    print("model loaded from", model_file)
else:
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    print("custom model not found, 'kogpt2-base-v2' model will be used")
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
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
num_training_steps = epochs * len(train_dataloader)
num_warmup_steps = int(num_training_steps * warmup_ratio)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


sample_dataset = next(iter(train_dataloader))
sample_dataset = {k: v.to(device) for k, v in sample_dataset.items()}
ic(sample_dataset["input_ids"][0])
ic(sample_dataset["masks"][0])
ic(sample_dataset["label_ids"][0])
ic(tokenizer.decode(sample_dataset["input_ids"][0]))
ic(tokenizer.decode(sample_dataset["masks"][0]))
masked_inputs = torch.where(sample_dataset["masks"][0] == 1, sample_dataset["input_ids"][0], 0)
ic(masked_inputs)
ic(tokenizer.decode(masked_inputs))
ic(tokenizer.decode(sample_dataset["label_ids"][0]))
ic(sample_dataset["input_ids"].shape)

outputs = model(sample_dataset["input_ids"], return_dict=True)
outputs = outputs.logits
masks_3d = (
    sample_dataset["masks"].unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
)
masks_out = torch.where(masks_3d == 1, outputs, negative * torch.ones_like(outputs))
loss = loss_fn(masks_out.transpose(1, 2), sample_dataset["label_ids"])
loss = loss.sum() / sample_dataset["masks"].sum()
ic(sample_dataset["masks"])
ic(sample_dataset["masks"].sum())
ic(loss)
ic(outputs.shape)


torch.manual_seed(seed=42)

batch_size = 1
max_len = 12
num_classes = 8

inputs = torch.zeros(size=(batch_size, max_len), dtype=torch.long).to(device)
inputs[0, 2:5] = torch.tensor([3, 4, 5])
ic(inputs)
masks = torch.zeros(size=(batch_size, max_len), dtype=torch.long).to(device)
masks[0, 2:5] = 1
ic(masks)
labels = torch.randint(low=0, high=num_classes, size=(batch_size, max_len), dtype=torch.long).to(
    device
)
ic(labels)
outputs = model(inputs, return_dict=True).logits
outputs = outputs[:, :, :num_classes]
ic(outputs.shape)
mask_3d = masks.unsqueeze(dim=2)
ic(mask_3d)
mask_3d = mask_3d.repeat_interleave(repeats=outputs.shape[2], dim=2)
ic(mask_3d)
ic(outputs[0, 0, :])
ic(mask_3d.sum())
mask_out = torch.where(mask_3d == 1, outputs, negative * torch.ones_like(outputs))
ic(mask_out)
ic(mask_out.shape)
ic(mask_out.transpose(1, 2).shape)
ic(labels.shape)
loss = loss_fn(mask_out.transpose(1, 2), labels)
ic(loss)


output = [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544]
output = [-1e20] * 10
output[1] = 0.805
output = torch.tensor(output, dtype=torch.float32)
target = torch.tensor([1], dtype=torch.long)
loss1 = torch.log(torch.sum(torch.exp(output))) - output[target[0]]
output = [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]
output = torch.tensor(output, dtype=torch.float32)
target = torch.tensor([5], dtype=torch.long)
loss2 = torch.log(torch.sum(torch.exp(output))) - output[target[0]]
ic(loss1)
ic(loss2)
ic((loss1 + loss2) / 2)


# ic| inputs: tensor([[0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
# ic| masks: tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
# ic| labels: tensor([[6, 3, 4, 6, 2, 7, 4, 4, 6, 1, 2, 6]], device='cuda:0')
outputs[:, :, :] = -1e20
ic(outputs.shape)
outputs[0, 2, 4] = torch.tensor(1.5)
outputs[0, 3, 6] = torch.tensor(6.5)
outputs[0, 4, 2] = torch.tensor(2.8)
ic(outputs[0, 2, :])
ic(outputs)

mask_3d = masks.unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
ic(mask_3d)
mask_out = torch.where(mask_3d == 1, outputs, negative * torch.ones_like(outputs))
ic(mask_out)
loss = loss_fn(mask_out.transpose(1, 2), labels)
ic(loss)

ic(mask_out.shape)
mask_out[0, 0, 6] = torch.tensor(1.5)
ic(mask_out[0, 0, :])
ic(labels[0, 0])
loss1 = loss_fn(mask_out[0, 0], labels[0, 0])
ic(loss1)
