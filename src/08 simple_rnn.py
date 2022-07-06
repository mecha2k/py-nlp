import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=4, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_classes = num_classes
        self.n_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        return self.activation(self.generator(x[:, -1]))


if __name__ == "__main__":
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
