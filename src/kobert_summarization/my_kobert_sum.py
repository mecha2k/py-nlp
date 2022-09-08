import torch
import torch.nn as nn
import transformers
import spacy
import warnings
import logging
import os

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.data.metrics import acc_and_f1
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from tensorflow.keras.utils import pad_sequences
from datasets import load_metric
from argparse import Namespace


logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = load_dataset("squad", split="train")
            self.val_dataset = load_dataset("squad", split="validation")

        if stage == "test" or stage is None:
            self.test_dataset = load_dataset("squad", split="validation")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        pass


class SimpleLinearClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, masks):
        x = self.linear(x).squeeze(-1)
        sentence_scores = x * masks.float()
        sentence_scores[sentence_scores == 0] = -1e10
        return sentence_scores


class KobertSummarization(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        self.model = AutoModel.from_pretrained("monologg/kobert")
        self.classifier = SimpleLinearClassifier(self.model.config.hidden_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = load_metric("squad")

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["start_positions"],
            batch["end_positions"],
        )
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["start_positions"],
            batch["end_positions"],
        )
        loss = outputs[0]
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["start_positions"],
            batch["end_positions"],
        )
        loss = outputs[0]
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def get_predictions(self, model, dataloader, compute_metrics=False):
        all_start_logits = []
        all_end_logits = []
        all_attention_masks = []
        all_input_ids = []
        all_contexts = []
        all_questions = []
        all_answers = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = model(batch["input_ids"], batch[""])


if __name__ == "__main__":
    pass
