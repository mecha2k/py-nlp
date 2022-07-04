import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([2.3, 3.4])
print(x)

x = x.new_ones(2, 3, dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)
print(x.size())

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x)
print(x.dtype)

print(x.short())
print(x.long())
print(x.int())

x = torch.IntTensor([[1, 2], [3, 4]])
print(x)
print(x.dtype)

print(x.float())
print(x.double())
print(x.half()) # half precision

x = torch.randn(1)
print(x)
print(x.item())
print(x.dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

y = torch.ones_like(x, device=device)
print(y)
print(y.device)

x = x.to(device)
z = x + y
print(z)

x = torch.Tensor([0.1])
print(x)
print(x.ndim)
print(x.shape)

x = torch.Tensor([1, 2, 3])
print(x)
print(x.ndim)
print(x.shape)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print(x.ndim)
print(x.shape)

x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print(x)
print(x.ndim)
print(x.shape)