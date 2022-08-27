import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import warnings
import logging

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import Namespace


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=Warning)
transformers.logging.set_verbosity_error()


class CnnDataModule(LightningDataModule):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.datasets = dict()

    def prepare_data(self) -> None:
        datasets = dict()
        data_types = ["train", "val", "test"]
        for data_type in data_types:
            inferred_type = "txt"
            dataset_files = glob.glob(
                os.path.join(hparams.data_path, "*" + data_type + ".*." + inferred_type)
            )
            datasets[data_type] = FSDataset(dataset_files, verbose=True)
        self.datasets = datasets

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass


class ExtractiveSummarization(LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams["model_name"])
        self.model = AutoModel.from_pretrained(self.hparams["model_name"])
        self.model.eval()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.hparams["learning_rate"])

    def forward(
        self,
        word_vectors: torch.Tensor,
        sent_rep_token_ids: torch.Tensor,
        sent_rep_mask: torch.Tensor,
    ) -> torch.Tensor:
        output_vectors, output_masks = [], []
        sents_vec = word_vectors[
            torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
        ]
        sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
        output_vectors.append(sents_vec)
        output_masks.append(sent_rep_mask)

        return torch.cat(output_vectors, 1), torch.cat(output_masks, 1)

    def training_step(self, batch, batch_idx) -> dict:
        word_vectors, sent_rep_token_ids, sent_rep_mask = batch
        output_vectors, output_masks = self.forward(word_vectors, sent_rep_token_ids, sent_rep_mask)
        loss = self.loss_fn(output_vectors, sent_rep_token_ids)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx) -> dict:
        word_vectors, sent_rep_token_ids, sent_rep_mask = batch
        return word_vectors, sent_rep_token_ids, sent_rep_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    hparams = Namespace(
        data_dir="../data/cnn_daily/cnn_dm/json.gz",
        model_name="bert-base-uncased",
        learning_rate=1e-5,
        batch_size=32,
        num_epochs=1,
        num_workers=0,
        max_seq_length=512,
        max_summary_length=64,
    )

    cnn_dm = CnnDataModule(data_dir=hparams.data_dir)
    model = ExtractiveSummarization(hparams)
    trainer = Trainer(
        max_epochs=1,
        max_steps=10,
        accelerator="auto",
        devices="auto",
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, datamodule=cnn_dm)
