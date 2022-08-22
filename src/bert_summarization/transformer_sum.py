import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers
import glob
import logging
import warnings
import os

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.metrics import acc_and_f1
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import Namespace

from dataset_proc import SentencesProcessor, FSDataset, FSIterableDataset, pad_batch_collate
from helpers import load_json, generic_configure_optimizers


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=Warning)
transformers.logging.set_verbosity_error()


class Pooling(nn.Module):
    def __init__(self, sent_rep_tokens=True, mean_tokens=False, max_tokens=False):
        super().__init__()
        self.sent_rep_tokens = sent_rep_tokens
        self.mean_tokens = mean_tokens
        self.max_tokens = max_tokens

    def forward(
        self,
        word_vectors=None,
        sent_rep_token_ids=None,
        sent_rep_mask=None,
        sent_lengths=None,
        sent_lengths_mask=None,
    ):
        output_vectors, output_masks = [], []

        if self.sent_rep_tokens:
            sents_vec = word_vectors[
                torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
            ]
            sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_rep_mask)

        if self.mean_tokens or self.max_tokens:
            batch_sequences = [
                torch.split(word_vectors[idx], seg) for idx, seg in enumerate(sent_lengths)
            ]
            sents_list = [
                torch.stack(
                    [
                        (
                            (sequence.sum(dim=0) / (sequence != 0).sum(dim=0))
                            if self.mean_tokens
                            else torch.max(sequence, 0)[0]
                        )
                        if ((sequence != 0).sum() != 0)
                        else word_vectors[0, 0].float()
                        for sequence in sequences
                    ],
                    dim=0,
                )
                for sequences in batch_sequences  # for all the sentences in each batch
            ]
            sents_vec = torch.stack(sents_list, dim=0)
            sents_vec = sents_vec * sent_lengths_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_lengths_mask)

        return torch.cat(output_vectors, 1), torch.cat(output_masks, 1)


class SimpleLinearClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        x = self.linear(x).squeeze(-1)
        sentence_scores = x * mask.float()
        sentence_scores[sentence_scores == 0] = -1e10
        return sentence_scores


class ExtractiveSummarization(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.forward_modify_inputs_callback = None

        model_config = AutoConfig.from_pretrained(
            self.hparams.model_name, gradient_checkpointing=self.hparams.gradient_checkpointing
        )
        self.model = AutoModel.from_config(model_config)

        if (
            any(x in hparams.model_name for x in ["roberta", "distil", "longformer"])
        ) and not hparams.no_use_token_type_ids:
            logger.warning(
                "You are using a %s model but did not set --no_use_token_type_ids. This model does not support "
                + "`token_type_ids` so this option has been automatically enabled.",
                hparams.model_type,
            )
            self.hparams.no_use_token_type_ids = True

        self.pooling = Pooling(sent_rep_tokens=True, mean_tokens=False, max_tokens=False)
        self.encoder = SimpleLinearClassifier(self.model.config.hidden_size)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.model_name, use_fast=hparams.tokenizer_use_fast
        )

        self.datasets = None
        self.dataloaders = None
        self.rouge_metrics = None
        self.rouge_scorer = None
        self.pad_batch_collate = pad_batch_collate

    def prepare_data(self):
        datasets = dict()
        data_types = ["train", "val", "test"]
        for data_type in data_types:
            inferred_type = "txt"
            dataset_files = glob.glob(
                os.path.join(hparams.data_path, "*" + data_type + ".*." + inferred_type)
            )
            datasets[data_type] = FSDataset(dataset_files, verbose=True)
        self.datasets = datasets

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            logger.info("Loading `word_embedding_model` pre-trained weights.")
            self.model = AutoModel.from_pretrained(
                self.hparams.model_name, config=self.model.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        sent_lengths_mask=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.hparams.use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.model(**inputs, **kwargs)
        sents_vector, mask = self.pooling(
            word_vectors=outputs[0],
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
            sent_lengths=sent_lengths,
            sent_lengths_mask=sent_lengths_mask,
        )
        sent_scores = self.encoder(sents_vector, mask)
        return sent_scores, mask

    def compute_loss(self, outputs, labels, mask):
        try:
            loss = self.loss_fn(outputs, labels.float())
        except ValueError as e:
            logger.error(e)
            logger.error(
                "Details about above error:\n1. outputs=%s\n2. labels.float()=%s",
                outputs,
                labels.float(),
            )
            sys.exit(1)

        # set all padding values to zero
        loss = loss * mask.float()
        # add up all the loss values for each sequence (including padding because
        # padding values are zero and thus will have no effect)
        sum_loss_per_sequence = loss.sum(dim=1)
        # count the number of losses that are not padding per sequence
        num_not_padded_per_sequence = mask.sum(dim=1).float()
        # find the average loss per sequence
        average_per_sequence = sum_loss_per_sequence / num_not_padded_per_sequence
        # get the sum of the average loss per sequence
        sum_avg_seq_loss = average_per_sequence.sum()  # sum_average_per_sequence
        # get the mean of `average_per_sequence`
        batch_size = average_per_sequence.size(0)
        mean_avg_seq_loss = sum_avg_seq_loss / batch_size

        # calculate the sum of all the loss values for each sequence
        total_loss = sum_loss_per_sequence.sum()
        # count the total number of losses that are not padding
        total_num_not_padded = num_not_padded_per_sequence.sum().float()
        # average loss
        average_loss = total_loss / total_num_not_padded
        # total loss normalized by batch size
        total_norm_batch_loss = total_loss / batch_size
        return (
            total_loss,
            total_norm_batch_loss,
            sum_avg_seq_loss,
            mean_avg_seq_loss,
            average_loss,
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        del batch["labels"]

        outputs, mask = self.forward(**batch)

        (
            loss_total,
            loss_total_norm_batch,
            loss_avg_seq_sum,
            loss_avg_seq_mean,
            loss_avg,
        ) = self.compute_loss(outputs, labels, mask)
        loss_dict = {
            "train_loss_total": loss_total,
            "train_loss_total_norm_batch": loss_total_norm_batch,
            "train_loss_avg_seq_sum": loss_avg_seq_sum,
            "train_loss_avg_seq_mean": loss_avg_seq_mean,
            "train_loss_avg": loss_avg,
        }
        for name, value in loss_dict.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        return loss_dict["train_" + self.hparams.loss_key]

    def configure_optimizers(self):
        return generic_configure_optimizers(
            self.hparams, self.train_dataloader(), self.named_parameters()
        )


if __name__ == "__main__":
    hparams = Namespace(
        max_epochs=10,
        batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        optimizer_type="adam",
        use_scheduler=False,
        loss_key="loss_avg",
        max_steps=1000,
        model_name="bert-base-uncased",
        model_type="bert",
        use_token_type_ids=False,
        tokenizer_use_fast=True,
        gradient_checkpointing=False,
        data_path="../data/cnn_daily/cnn_dm/json.gz",
        data_type="txt",
        dataloader_type="map",
        create_token_type_ids="binary",
        max_seq_length=512,
        accumulate_grad_batches=1,
    )
    print("num_epochs : ", getattr(hparams, "epochs", 1))

    model_checkpoint = ModelCheckpoint(
        dirpath="../data/cnn_daily/checkpoints",
        monitor="val_loss",
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [model_checkpoint, lr_monitor]

    model = ExtractiveSummarization(hparams=hparams)
    trainer = Trainer(
        default_root_dir="../data/cnn_daily",
        max_steps=hparams.max_steps,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
    )

    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
    trainer.fit(model)
