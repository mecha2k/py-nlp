import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt


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
print(probabilities)
print(loss)
