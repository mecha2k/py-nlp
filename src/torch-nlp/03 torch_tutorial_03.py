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
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.current_stream())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

plt.style.use("seaborn")
plt.rcParams["font.size"] = 16
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

bce_loss = nn.BCELoss()
probabilities = nn.Sigmoid()(torch.randn(5, 1, requires_grad=True))
targets = torch.tensor([1, 0, 0, 1, 0], dtype=torch.float32).view(-1, 1)
loss = bce_loss(probabilities, targets)
loss.backward()
print(probabilities)
print(loss)

loss_fn = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True).to(device)
targets = torch.tensor([1, 3, 2], dtype=torch.int64).to(device)
loss = loss_fn(outputs, targets)
loss.backward()
print(outputs)
print(loss)


import re

text = "I am a NLPer>?,!"
text = re.sub(r"([.,!?])", r" \1 ", text)
print(text)
