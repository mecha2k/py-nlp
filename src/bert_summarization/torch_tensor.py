import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.datasets import MNIST
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from icecream import ic


torch.manual_seed(42)


def forward(
    word_vectors=None,
    sent_rep_token_ids=None,
    sent_rep_mask=None,
):
    output_vectors, output_masks = [], []
    sents_vec = word_vectors[torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids]
    sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
    output_vectors.append(sents_vec)
    output_masks.append(sent_rep_mask)

    return torch.cat(output_vectors, 1), torch.cat(output_masks, 1)


batch_size = 2
max_seq_length = 16
hidden_size = 4
sent_size = 4

word_vectors = torch.randn(size=(batch_size, max_seq_length, hidden_size), dtype=torch.float32)
sent_lengths = torch.randint(low=1, high=sent_size, size=(batch_size, sent_size), dtype=torch.long)
for snl in sent_lengths:
    ic(snl)
    if snl.sum() != max_seq_length:
        snl[-1] += max_seq_length - snl.sum()
sent_lengths_mask = torch.randint(low=0, high=1, size=(batch_size, sent_size), dtype=torch.bool)
sent_rep_mask = torch.randint(low=0, high=1, size=(batch_size, sent_size), dtype=torch.bool)
sent_rep_token_ids = torch.randint(
    low=0, high=max_seq_length, size=(batch_size, sent_size), dtype=torch.long
)

ic(word_vectors)
ic(word_vectors.size())
ic(sent_lengths)
ic(sent_lengths_mask)
ic(sent_rep_mask)
ic(sent_rep_token_ids)

sents_vec = word_vectors[torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids]
ic(word_vectors[[[0], [1]], [1, 2]])
ic(torch.arange(word_vectors.size(0)).unsqueeze(1))
ic(word_vectors[torch.arange(word_vectors.size(0)).unsqueeze(1), [1, 2]])
ic(sents_vec)
ic(sent_rep_mask.size())
ic(sent_rep_mask[:, :, None])
ic(sent_rep_mask.unsqueeze(dim=2).float())
ic(sent_rep_mask.unsqueeze(dim=2).float().size())
sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
ic(sents_vec)

for idx, seg in enumerate(sent_lengths):
    ic(idx, seg)
    ic(word_vectors[idx])
    splitted = torch.split(word_vectors[idx], [4, 1, max_seq_length - 4 - 1])
    ic(splitted)
    splitted = torch.split(word_vectors[idx], seg.tolist())
    ic(len(word_vectors[idx]))
    ic(splitted[0])
    ic(len(splitted))
    break


batch_sequences = [
    torch.split(word_vectors[idx], seg.tolist()) for idx, seg in enumerate(sent_lengths)
]
ic(len(batch_sequences))


pooling = forward(word_vectors, sent_rep_token_ids, sent_rep_mask)
ic(pooling)


batch = {"source": 1, "target": 2, "sent_rep_token_ids": 3, "sent_rep_mask": 4}
if "source" and "target" in batch.keys():
    print("source", batch["source"])
    print("target", batch["target"])


import pyrouge

rouge = pyrouge.Rouge155()
