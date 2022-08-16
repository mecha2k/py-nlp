from lib2to3.pgen2 import grammar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.metrics import acc_and_f1
from argparse import Namespace


logger = logging.getLogger(__name__)

args = Namespace(
    epochs=10,
    batch_size=1,
    learning_rate=0.1,
    model_name="bert-base-uncased",
    model_type="bert",
    no_use_token_type_ids=True,
    tokenizer_use_fast=True,
    gradient_checkpointing=False,
    tokenizer_no_use_fast=False,
    data_type="none",
)
print(getattr(args, "epochs", 1))


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


class ExtractiveSummarization(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.forward_modify_inputs_callback = None

        model_config = AutoConfig.from_pretrained(
            hparams.model_name, gradient_checkpointing=hparams.gradient_checkpointing
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

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    # def training_step(self, batch, batch_idx):
    #     input_ids, attention_mask, labels = batch
    #     input_ids = input_ids.to(self.device)
    #     attention_mask = attention_mask.to(self.device)
    #     labels = labels.to(self.device)
    #
    #     outputs = self.forward(input_ids, attention_mask)
    #     loss = self.loss_fn(outputs, labels)
    #
    #     tensorboard_logs = {"train_loss": loss}
    #     return {"loss": loss, "log": tensorboard_logs}
    #
    # def validation_step(self, batch, batch_idx):
    #     input_ids, attention_mask, labels = batch
    #     input_ids = input_ids.to(self.device)
    #     attention_mask = attention_mask.to(self.device)
    #     labels = labels.to(self.device)
    #
    #     outputs = self.forward(input_ids, attention_mask)
    #     loss = self.loss_fn(outputs, labels)
    #
    #     return {"val_loss": loss}
    #
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     tensorboard_logs = {"val_loss": avg_loss}
    #     return {"val_loss": avg_loss, "log": tensorboard_logs}
