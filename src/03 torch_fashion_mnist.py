import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


fmnist_train = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=1.0)]),
)
fmnist_test = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=1.0)]),
)

train_loader = DataLoader(fmnist_train, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=0)

labels_map = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(images[i].squeeze().numpy(), cmap="gray")
    plt.title(labels_map[labels[i].item()], fontsize=18)
plt.savefig("images/fashion_mnist_dataset", bbox_inches="tight")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


epochs = 1
batch_size = 32
learning_rate = 0.001

model = NeuralNetwork().to(device)
print(model)

params = list(model.parameters())
print(len(params))
print(params[0].shape)

x = torch.randn(1, 1, 28, 28).to(device)
y = model(x)
print(y.shape)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}\n-------------------------------")
#     train_loop(train_loader, model, loss_fn, optimizer)
#     test_loop(test_loader, model, loss_fn)
# print("Done!")
