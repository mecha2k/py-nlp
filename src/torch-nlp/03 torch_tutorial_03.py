import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")
if device == "cuda":
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.current_stream())

plt.style.use("seaborn")
plt.rcParams["font.size"] = 36
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

bce_loss = nn.BCELoss()
probabilities = nn.Sigmoid()(torch.randn(5, 1, requires_grad=True))
targets = torch.tensor([1, 0, 0, 1, 0], dtype=torch.float32).view(-1, 1)
loss = bce_loss(probabilities, targets)
loss.backward()
print(probabilities)
print(loss)


loss_fn = nn.MSELoss()
outputs = torch.tensor([[0, 1, 1, 1, 2], [1, 1, 0, 2, 1]], dtype=torch.float32, requires_grad=True)
targets = torch.tensor([[1, 0, 0, 1, 2], [0, 2, 1, 1, 0]], dtype=torch.float32)
loss = loss_fn(outputs, targets)
print("MSELoss : ", loss.item())


loss_fn = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True).to(device)
targets = torch.tensor([1, 3, 2], dtype=torch.int64).to(device)
loss = loss_fn(outputs, targets)
loss.backward()
print(outputs)
print("CrossEntropyLoss : ", loss.item())


output = torch.tensor(
    [
        [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544],
        [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332],
    ],
    dtype=torch.float32,
)
target = torch.tensor([1, 5], dtype=torch.long)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print("CrossEntropyLoss : ", loss.item())

output = [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544]
target = [1]
loss1 = np.log(sum(np.exp(output))) - output[target[0]]
output = [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]
target = [5]
loss2 = np.log(sum(np.exp(output))) - output[target[0]]
print("CrossEntropyLoss : ", (loss1 + loss2) / 2)

import re
import string
from collections import Counter

text = "I am a NLPer>?,!"
text = re.sub(r"([.,!?])", r" \1 ", text)
print(text)

for i in range(3):
    print(np.random.normal(loc=(3, 3), scale=1))

word_counts = Counter()
title = "bank of montreal to acquire indiana based institution"
for token in title.split(" "):
    if token not in string.punctuation:
        word_counts[token] += 1
        print(token)
print(word_counts)
print(string.punctuation)

word_counts = Counter(title.split(" "))
print(word_counts)
