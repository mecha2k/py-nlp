import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import pyrouge
import warnings
import logging
import os

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.data.metrics import acc_and_f1
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge_score.rouge_scorer import RougeScorer
from collections import OrderedDict
from argparse import Namespace
from time import strftime, localtime


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=Warning)
transformers.logging.set_verbosity_error()


def _get_ngrams(n_gram: int, text: list) -> set:
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n_gram
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n_gram]))
    return ngram_set


def block_trigrams(candidate: str, prediction: list) -> bool:
    tri_c = _get_ngrams(3, candidate.split())
    for s in prediction:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False


def test_rouge(candidate_file, reference_file):
    candidates = [line.strip() for line in open(candidate_file, encoding="utf-8")]
    references = [line.strip() for line in open(reference_file, encoding="utf-8")]
    assert len(candidates) == len(references)

    rouge_dir = "../data/cnn_daily/rouge"
    os.makedirs(rouge_dir, exist_ok=True)
    current_time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    rouge_dir = os.path.join(rouge_dir, f"rouge-{current_time}")
    os.makedirs(rouge_dir, exist_ok=True)
    os.makedirs(rouge_dir + "/candidate", exist_ok=True)
    os.makedirs(rouge_dir + "/reference", exist_ok=True)

    try:
        for i in range(len(candidates)):
            if len(references[i]) < 1:
                continue
            with open(rouge_dir + f"/candidate/candidate.{i}.txt", "w", encoding="utf-8") as file:
                file.write(candidates[i].replace("<q>", "\n"))
            with open(rouge_dir + f"/reference/reference.{i}.txt", "w", encoding="utf-8") as file:
                file.write(references[i].replace("<q>", "\n"))

        rouge = pyrouge.Rouge155()
        rouge.mod_dir = rouge_dir + "/reference/"
        rouge.sys_dir = rouge_dir + "/candidate/"
        rouge.mod_file_pattern = "reference.#ID#.txt"
        rouge.sys_file_pattern = r"candidate.(\d+).txt"
        rouge_results = rouge.convert_and_evaluate()
        print(rouge_results)
        results = rouge.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(rouge_dir):
            shutil.rmtree(rouge_dir)
    return results


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

        sources, targets = None, None
        if "source" and "target" in batch[0].keys():
            sources = [item["source"] for item in batch]
            targets = [item["target"] for item in batch]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sent_rep_token_ids": torch.tensor(sent_rep_token_ids, dtype=torch.long),
            "sent_rep_masks": torch.tensor(sent_rep_masks, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sources": sources,
            "targets": targets,
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

        self.rouge_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.rouge_scorer = RougeScorer(self.rouge_metrics, use_stemmer=True)

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

    @staticmethod
    def dict_keys(batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        sent_rep_token_ids = batch["sent_rep_token_ids"]
        sent_rep_masks = batch["sent_rep_masks"]
        labels = batch["labels"]

        sources, targets = None, None
        if "sources" and "targets" in batch.keys():
            sources = batch["sources"]
            targets = batch["targets"]

        return (
            input_ids,
            attention_mask,
            token_type_ids,
            sent_rep_token_ids,
            sent_rep_masks,
            labels,
            sources,
            targets,
        )

    def training_step(self, batch, batch_idx) -> dict:
        (
            input_ids,
            attention_mask,
            token_type_ids,
            sent_rep_token_ids,
            sent_rep_masks,
            labels,
            sources,
            targets,
        ) = self.dict_keys(batch)

        outputs = self(
            input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
        )
        loss = self.compute_loss(outputs, labels, sent_rep_masks)
        self.log("train_loss", loss[0])

        return {"loss": loss[0]}

    def validation_step(self, batch, batch_idx):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            sent_rep_token_ids,
            sent_rep_masks,
            labels,
            sources,
            targets,
        ) = self.dict_keys(batch)

        outputs = self(
            input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
        )
        loss = self.compute_loss(outputs, labels, sent_rep_masks)
        self.log("val_loss", loss[0], prog_bar=True)

    def test_step(self, batch, batch_idx):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            sent_rep_token_ids,
            sent_rep_masks,
            labels,
            sources,
            targets,
        ) = self.dict_keys(batch)

        outputs = self(
            input_ids, attention_mask, token_type_ids, sent_rep_token_ids, sent_rep_masks
        )
        outputs = torch.sigmoid(outputs)

        y_pred = outputs.clone().detach()
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        y_pred = torch.flatten(y_pred).cpu().numpy()
        y_true = torch.flatten(labels).cpu().numpy()
        result = acc_and_f1(y_pred, y_true)

        sorted_ids = torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()

        predictions = []
        for idx, (source, source_ids, target) in enumerate(zip(sources, sorted_ids, targets)):
            current_prediction = []
            for sent_idx, i in enumerate(source_ids):
                if i >= len(source):
                    logger.debug(
                        "Only %i examples selected from document %i in batch %i. This is likely because some sentences "
                        + "received ranks so small they rounded to zero and a padding 'sentence' was randomly chosen.",
                        sent_idx + 1,
                        idx,
                        batch_idx,
                    )
                    continue

                candidate = source[i].strip()
                if not block_trigrams(candidate, current_prediction):
                    current_prediction.append(candidate)

                if len(current_prediction) >= self.hparams.top_k_sentences:
                    break

            current_prediction = "<q>".join(current_prediction)
            predictions.append(current_prediction)

        with open("../data/cnn_daily/save_gold.txt", "w", encoding="utf-8") as save_gold, open(
            "../data/cnn_daily/save_pred.txt", "w", encoding="utf-8"
        ) as save_pred:
            for target in targets:
                save_gold.write(target.strip() + "\n")
            for prediction in predictions:
                save_pred.write(prediction.strip() + "\n")

        return OrderedDict(
            {
                "acc": torch.tensor(result["acc"]),
                "f1": torch.tensor(result["f1"]),
                "acc_and_f1": torch.tensor(result["acc_and_f1"]),
            }
        )

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["f1"] for x in outputs]).mean()
        avg_acc_and_f1 = torch.stack([x["acc_and_f1"] for x in outputs]).mean()

        test_rouge("../data/cnn_daily/save_pred.txt", "../data/cnn_daily/save_gold.txt")

        # rouge_scores_log = {}
        #
        # if self.hparams.test_use_pyrouge:
        #     test_rouge("tmp", "save_pred.txt", "save_gold.txt")
        # else:
        #     aggregator = scoring.BootstrapAggregator()
        #
        #     # In `outputs` there is an entry for each batch that was passwed through the
        #     # `test_step()` function. For each batch a list containing the rouge scores
        #     # for each example exists under the key "rouge_scores" in `batch_list`. Thus,
        #     # the below list comprehension loops through the list of outputs and grabs the
        #     # items stored under the "rouge_scores" key. Then it flattens the list of lists
        #     # to a list of rouge score objects that can be added to the `aggregator`.
        #     rouge_scores_list = [
        #         rouge_score_set
        #         for batch_list in outputs
        #         for rouge_score_set in batch_list["rouge_scores"]
        #     ]
        #     for score in rouge_scores_list:
        #         aggregator.add_scores(score)
        #     # The aggregator returns a dictionary with keys coresponding to the rouge metric
        #     # and values that are `AggregateScore` objects. Each `AggregateScore` object is a
        #     # named tuple with a low, mid, and high value. Each value is a `Score` object, which
        #     # is also a named tuple, that contains the precision, recall, and fmeasure values.
        #     # For more info see the source code: https://github.com/google-research/google-research/blob/master/rouge/scoring.py  # noqa: E501
        #     rouge_result = aggregator.aggregate()
        #
        #     for metric, value in rouge_result.items():
        #         rouge_scores_log[metric + "-precision"] = value.mid.precision
        #         rouge_scores_log[metric + "-recall"] = value.mid.recall
        #         rouge_scores_log[metric + "-fmeasure"] = value.mid.fmeasure
        #
        # # Generate logs
        # loss_dict = {
        #     "test_acc": avg_test_acc,
        #     "test_f1": avg_test_f1,
        #     "avg_test_acc_and_f1": avg_test_acc_and_f1,
        # }
        #
        # for name, value in loss_dict.items():
        #     self.log(name, value, prog_bar=True, sync_dist=True)
        # for name, value in rouge_scores_log.items():
        #     self.log(name, value, prog_bar=False, sync_dist=True)
        #
        # avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        # self.log("test_loss", avg_loss)
        # self.log("test_acc", torch.stack([x["test_acc"] for x in outputs]).mean())
        # self.log("test_f1", torch.stack([x["test_f1"] for x in outputs]).mean())
        # self.log("test_acc_and_f1", torch.stack([x["test_acc_and_f1"] for x in outputs]).mean())
        #
        # return {"test_loss": avg_loss}

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
        load_from_checkpoint=False,
        model_name="prajjwal1/bert-small",  # "bert-base-uncased"
        learning_rate=1e-5,
        batch_size=32,
        num_epochs=100,
        max_seq_len=512,
        weight_decay=0.01,
        top_k_sentences=3,
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
    trainer.test(model, datamodule=cnn_dm)
