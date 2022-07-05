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

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=1.0)]),
)
mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=1.0)]),
)

train_loader = DataLoader(mnist_train, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(mnist_test, batch_size=8, shuffle=False, num_workers=0)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)

image = torch.squeeze(images[0])
print(image.shape)
plt.imshow(image.numpy(), cmap="gray")
plt.savefig("images/mnist_image", dpi=200)

fig = plt.figure(figsize=(12, 6))
rows, cols = 2, 4
for i in range(1, rows * cols + 1):
    index = np.random.randint(0, len(mnist_train), size=1)[0]
    image, label = mnist_train[index]
    fig.add_subplot(rows, cols, i)
    plt.title(label, fontsize=18)
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")
plt.savefig("images/mnist_image")

x = torch.randn(128, 20)
model = nn.Linear(20, 10)
y = model(x)
print(y.shape)
print(y)

x = torch.randn(20, 16, 50, 100)
model = nn.Conv2d(16, 33, kernel_size=3, stride=2)
model = nn.Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))

y = model(x)
print(y.shape)

layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2).to(device)
print(layer)

weight = layer.weight
print("weight:", weight.shape)

print(images.shape)
print(images[0].shape)

image = images[0].squeeze()
x = image.unsqueeze(dim=0)
print("x:", x.shape)
y = layer(x)
print("y:", y.shape)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.axis("off")
plt.title("input", fontsize=18)
plt.imshow(x[0, :, :].detach().cpu().numpy(), cmap="gray")
plt.subplot(1, 3, 2)
plt.axis("off")
plt.title("output", fontsize=18)
plt.imshow(y[0, :, :].detach().cpu().numpy(), cmap="gray")
plt.subplot(1, 3, 3)
plt.axis("off")
plt.title("weight", fontsize=18)
plt.imshow(weight[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
plt.tight_layout()
plt.savefig("images/mnist_conv")
