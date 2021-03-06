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

pool = F.max_pool2d(y, kernel_size=2, stride=2)
print(pool.shape)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pool = pool(y)
print(pool.shape)
pool = pool.detach().numpy()
print(pool.shape)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("input", fontsize=18)
plt.imshow(x[0, :, :].detach().cpu().numpy(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("output", fontsize=18)
plt.imshow(pool[0, :, :], cmap="gray")
plt.tight_layout()
plt.savefig("images/mnist_pool")

with torch.no_grad():
    x = x.view(-1, 28 * 28)
    x = nn.Linear(784, 10)(x)
    x = F.softmax(x, dim=1)
print(x)
print(np.sum(x.numpy()))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=10, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x


model = Model()
print(list(model.children()))
print(list(model.modules()))

loss = nn.BCELoss()
loss = nn.CrossEntropyLoss()
loss = nn.MSELoss()


import torchmetrics as tm

preds = torch.randn(10, 5).softmax(dim=1)
preds_sum = torch.sum(preds, dim=1)
print(preds)
print(preds_sum)
print(preds.shape)
labels = torch.randint(high=5, size=(10,))
print(labels)

acc = tm.functional.accuracy(preds, labels)
print(acc)

metric = tm.Accuracy()

batch_size = 2
for i in range(batch_size):
    preds = torch.randn(10, 5).softmax(dim=1)
    labels = torch.randint(high=5, size=(10,))
    acc = metric(preds, labels)
acc = metric.compute()
print(acc)


X = torch.randn(200, 1) * 10
y = X + 3 * torch.randn(200, 1)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X.numpy(), y.numpy())


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
print(list(model.children()))
print(list(model.modules()))
print(model)
print(list(model.parameters()))

weight, bias = model.parameters()
w1, b1 = weight.data.numpy()[0][0], bias.data.numpy()[0]
print(w1, b1)

x1 = np.array([-30, 30])
y1 = w1 * x1 + b1
plt.plot(x1, y1, color="red")
plt.savefig("images/linear_regression")


loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item())

fig = plt.figure(figsize=(8, 6))
plt.plot(range(epochs), losses)
plt.savefig("images/linear_regression_loss")


weight, bias = model.parameters()
w1, b1 = weight.data.numpy()[0][0], bias.data.numpy()[0]
y1 = w1 * x1 + b1

fig = plt.figure(figsize=(8, 6))
plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, color="red")
plt.savefig("images/linear_regression1")
