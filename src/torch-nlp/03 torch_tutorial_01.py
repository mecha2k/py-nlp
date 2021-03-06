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
print(x.half())  # half precision

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

x = torch.Tensor(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]
)
print(x)
print(x.ndim)
print(x.shape)


x = torch.randn(2, 3) * 2 - 1
print(x)
print(x.size())

x = torch.abs(x)
print(x)
print(torch.ceil(x))
print(torch.floor(x))
print(torch.round(x))
print(torch.sqrt(x))
print(torch.log(x))
print(torch.sin(x))
print(x)
print(torch.clamp(x, 0, 2))

print(torch.min(x))
print(torch.max(x))
print(torch.mean(x))
print(torch.std(x))
print(torch.sum(x))
print(torch.prod(x))
print(torch.round(x))

x = torch.IntTensor([1, 2, 3, 4, 5])
print(torch.unique(x))
print(torch.unique(x, sorted=True))
print(torch.unique(x, return_inverse=True))
print(torch.unique(x, return_inverse=True, return_counts=True))


x = torch.randn(2, 3)  # 2x3 matrix with normal distribution
print(x)
x = torch.rand(2, 3)  # uniform distribution
print(x)
print(x.max(dim=0))
print(x.max(dim=1))
print(torch.max(x, dim=0))

x = torch.rand(2, 3)
y = torch.rand(2, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(2, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x / y)
print(torch.div(x, y))
print(x.div(y))

y = torch.rand(3, 3)
print(torch.matmul(x, y))

z = torch.mm(x, y)
print(torch.svd(z))


x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, 1])
print(x[0, 1:])
print(x[:, 1:])
print(x[:, 1:2])
print(x[:, 1:2:1])

x = torch.rand(3, 4)
print(x)
print(x.view(1, 12))
print(x.view(2, -1))
print(x.view(-1, 3))

x = torch.rand(1)
print(x.item())

x = torch.rand(1, 3, 4)
print(x.size())
print(x.shape)
print(x.squeeze().size())

x = torch.rand(3, 4, 1)
print(x.size())
print(x.shape)
print(x.squeeze().size())

x = x.squeeze()
print(x.size())
x = x.unsqueeze(dim=1)
print(x.size())

x = torch.FloatTensor([1, 2])
y = torch.FloatTensor([3, 4])
z = torch.FloatTensor([5, 6])
print(torch.stack([x, y, z]))


x = torch.randint(0, 2, size=(1, 2, 3))
y = torch.randint(0, 2, size=(1, 2, 3))
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))
print(torch.cat([x, y], dim=2))
print(torch.cat([x, y], dim=0).size())
print(torch.cat([x, y], dim=1).size())
print(torch.cat([x, y], dim=2).size())

x = torch.rand(3, 6)
x1, x2, x3 = torch.chunk(x, chunks=3, dim=0)
print(x1)
print(x2)
print(x3)
x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
print(x1)
print(x2)
print(x3)

x = torch.rand(3, 6)
x1, x2 = torch.split(x, split_size_or_sections=3, dim=1)
print(x)
print(x1)
print(x2)

x = torch.ones(7)
y = x.numpy()
print(x)
print(y)  # convert to numpy
print(x.add_(1))
print(np.add(y, 1, out=y))
print(torch.from_numpy(y))  # convert to torch

x = torch.randn(3, 3)
y = 3 * x
print(y.requires_grad)
print(y)
y.requires_grad_(True)
print(y.requires_grad)
print(y)
z = (y * y).sum()
print(y * y)
print(z.grad_fn)


x = torch.ones(3, 3, requires_grad=True)
print(x)

y = x + 5
print(y)

z = y * y
print(z)

out = z.mean()
print(out)
print(out.grad_fn)

out.backward()
print(x.grad)


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
print(v)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).any())

x = torch.ones(2, 2, requires_grad=True)
print(x.data)
print(x.grad)
print(x.grad_fn)

y = x + 2
z = y * y
out = z.sum()
print(out)

out.backward()
print(out.grad_fn)

print(x.data)
print(x.grad)
print(x.grad_fn)

print(y.data)
# print(y.grad)
print(y.grad_fn)

print(z.data)
# print(z.grad)
print(z.grad_fn)
