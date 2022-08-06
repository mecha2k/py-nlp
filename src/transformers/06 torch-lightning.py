import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer
from multiprocessing import freeze_support


class LitAutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = MNIST("../data/MNIST", download=True, transform=transforms.ToTensor())
    dataset = Subset(dataset, torch.arange(10000))
    train, valid = random_split(dataset=dataset, lengths=[9000, 1000])
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=64, shuffle=False, num_workers=0)

    model = LitAutoEncoder()
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
