import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn")
plt.rcParams["font.size"] = 18
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False


epochs = 30
batch_size = 1000
lr = 0.01
input_dim = 2


class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


def get_toy_data(batch_size, left_center=(3, 3), right_center=(3, -2)):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return np.array(x_data), y_targets


def visualize_results(model, X, y, title=None, epoch=None):
    y_pred = model(X)
    y_pred = (y_pred > 0.5).long().data.numpy().astype(np.int32)

    X = X.numpy()
    y = y.numpy().astype(np.int32)

    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        if y[i] == 0:
            marker = "o"
        else:
            marker = "p"
        if y_pred[i] == y[i]:
            color = ["dodgerblue"]
        else:
            color = ["coral"]
        plt.scatter(X[i][0], X[i][1], c=color, marker=marker, s=200)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    xlim = (min([x[0] for x in X]), max([x[0] for x in X]))
    ylim = (min([x[1] for x in X]), max([x[1] for x in X]))

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model(torch.tensor(xy, dtype=torch.float32)).detach().numpy().reshape(XX.shape)
    plt.contour(XX, YY, Z, colors="green", levels=[0.4, 0.5, 0.6], linestyles=["--", "-", "--"])
    plt.suptitle(title)
    plt.text(xlim[0], ylim[1], "Epoch = {}".format(str(epoch)))
    plt.savefig(f"images/super_{title}.png")


if __name__ == "__main__":
    X, y = get_toy_data(batch_size=batch_size)
    print(X.shape, y.shape)

    lx, rx, lc, rc = [], [], [], []
    for x_data, y_data in zip(X, y):
        if y_data == 0:
            lx.append(x_data)
            lc.append("green")
        else:
            rx.append(x_data)
            rc.append("blue")
    lx = np.stack(lx)
    rx = np.stack(rx)

    plt.figure(figsize=(5, 5))
    plt.scatter(lx[:, 0], lx[:, 1], color=lc, marker="*", s=100)
    plt.scatter(rx[:, 0], rx[:, 1], facecolor="white", edgecolor=rc, marker="o", s=100)
    plt.grid(True)
    plt.savefig("images/toy_data.png")

    model = Perceptron(input_dim=input_dim)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    visualize_results(model, X, y, title="initial", epoch=0)

    last = 0
    losses = []
    title = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_data, y_data = get_toy_data(batch_size=batch_size)
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)
        y_pred = model(x_data).squeeze()
        loss = bce_loss(y_pred, y_data)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        change = abs(last - loss_value)
        last = loss_value

    visualize_results(model, X, y, epoch=epochs, title="final")
