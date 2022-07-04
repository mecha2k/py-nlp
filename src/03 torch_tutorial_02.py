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
plt.rcParams["figure.dpi"] = 100

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
    plt.title(label)
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")
plt.savefig("images/mnist_image")
