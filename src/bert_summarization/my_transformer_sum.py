import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import warnings
import logging
import os

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from argparse import Namespace


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=Warning)
transformers.logging.set_verbosity_error()


class CnnDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.datasets = dict()

    def prepare_data(self) -> None:
        datasets = dict()
        data_types = ["train", "val", "test"]
        for data_type in data_types:
            datasets[data_type] = np.load(
                os.path.join(self.data_dir, f"datasets_{data_type}_small.npy"), allow_pickle=True
            )
        self.datasets = datasets

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            logger.info("Loading train data...")
        if stage == "test" or stage is None:
            logger.info("Loading test data...")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [[1] * len(input_id) for input_id in input_ids]
        token_type_ids = [item["token_type_ids"] for item in batch]
        sent_rep_token_ids = [item["sent_rep_token_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = pad_sequences(input_ids, maxlen=self.max_seq_len, padding="post")
        attention_mask = pad_sequences(attention_mask, maxlen=self.max_seq_len, padding="post")
        token_type_ids = pad_sequences(token_type_ids, maxlen=self.max_seq_len, padding="post")
        sent_rep_token_ids = pad_sequences(sent_rep_token_ids, padding="post", value=-1)
        labels = pad_sequences(labels, padding="post")

        sent_rep_masks = ~(sent_rep_token_ids == -1)
        sent_rep_token_ids[~sent_rep_masks] = 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sent_rep_token_ids": torch.tensor(sent_rep_token_ids, dtype=torch.long),
            "sent_rep_masks": torch.tensor(sent_rep_masks, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SimpleLinearClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, masks):
        x = self.linear(x).squeeze(-1)
        sentence_scores = x * masks.float()
        sentence_scores[sentence_scores == 0] = -1e10
        return sentence_scores


class ExtractiveSummarization(LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.hparams.model_name)
        self.classifier = SimpleLinearClassifier(self.model.config.hidden_size)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
    ) -> torch.Tensor:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        outputs = self.model(**inputs)
        hidden_states = outputs[0]
        sentence_vectors = hidden_states[
            torch.arange(hidden_states.size(0)).unsqueeze(dim=1), sent_rep_token_ids
        ]
        sentence_vectors = sentence_vectors * sent_rep_masks[:, :, None].float()
        sentence_scores = self.classifier(sentence_vectors, sent_rep_masks)

        return sentence_scores

    def compute_loss(self, outputs, labels, masks):
        loss = self.loss_fn(outputs, labels.float()) * masks.float()

        sum_loss_per_sequence = loss.sum(dim=1)
        num_not_padded_per_sequence = masks.sum(dim=1).float()
        average_per_sequence = sum_loss_per_sequence / num_not_padded_per_sequence

        sum_avg_seq_loss = average_per_sequence.sum()
        batch_size = average_per_sequence.size(0)
        mean_avg_seq_loss = sum_avg_seq_loss / batch_size

        total_loss = sum_loss_per_sequence.sum()
        total_num_not_padded = num_not_padded_per_sequence.sum().float()
        average_loss = total_loss / total_num_not_padded
        total_norm_batch_loss = total_loss / batch_size

        return (
            total_loss,
            total_norm_batch_loss,
            sum_avg_seq_loss,
            mean_avg_seq_loss,
            average_loss,
        )

    def training_step(self, batch, batch_idx) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        sent_rep_token_ids = batch["sent_rep_token_ids"]
        sent_rep_masks = batch["sent_rep_masks"]
        labels = batch["labels"]
        outputs = self(
            input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
        )
        loss = self.compute_loss(outputs, labels, sent_rep_masks)
        self.log("train_loss", loss[0])

        return {"loss": loss[0]}

    def validation_step(self, batch, batch_idx) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        sent_rep_token_ids = batch["sent_rep_token_ids"]
        sent_rep_masks = batch["sent_rep_masks"]
        labels = batch["labels"]
        outputs = self(
            input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
        )
        loss = self.compute_loss(outputs, labels, sent_rep_masks)
        self.log("val_loss", loss[0], prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_no_decay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        parameters = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(parameters, lr=self.hparams.learning_rate, eps=1e-8)


if __name__ == "__main__":
    hparams = Namespace(
        data_dir="../data/cnn_daily/cnn_dm/",
        model_dir="../data/cnn_daily/checkpoints/",
        model_file="../data/cnn_daily/checkpoints/my-bert-base-uncased.ckpt",
        load_from_checkpoint=True,
        model_name="bert-base-uncased",
        learning_rate=1e-5,
        batch_size=32,
        num_epochs=100,
        max_seq_len=512,
        weight_decay=0.01,
    )

    cnn_dm = CnnDataModule(
        data_dir=hparams.data_dir, batch_size=hparams.batch_size, max_seq_len=hparams.max_seq_len
    )

    if hparams.load_from_checkpoint:
        model = ExtractiveSummarization.load_from_checkpoint(hparams.model_file)
    else:
        model = ExtractiveSummarization(hparams=hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.model_dir,
        filename="my-bert-base-uncased-{epoch}",
        save_top_k=2,
        monitor="train_loss",
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        max_steps=1000,
        accelerator="auto",
        devices="auto",
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            TQDMProgressBar(refresh_rate=20),
        ],
    )

    trainer.fit(model, datamodule=cnn_dm)
