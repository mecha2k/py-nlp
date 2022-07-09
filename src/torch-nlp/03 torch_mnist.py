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

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=1.0),
        ]
    ),
)
mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=1.0),
        ]
    ),
)

epochs = 1
batch_size = 128
learning_rate = 0.001


train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0
)


images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(images[i].squeeze().numpy(), cmap="gray")
    plt.title(labels[i].item(), fontsize=18)
plt.savefig("images/mnist_dataset", bbox_inches="tight")


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


model = MnistModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")


with torch.no_grad():
    correct = 0
    for X, y in test_loader:
        pred = model(X.to(device))
        correct += (pred.argmax(dim=1) == y.to(device)).sum().item()
    print(f"accuracy: {correct / len(test_loader.dataset):>7f}")
