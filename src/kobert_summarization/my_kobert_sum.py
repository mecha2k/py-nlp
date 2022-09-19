import numpy as np
import pandas as pd
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_metric
from argparse import Namespace


logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


def _get_ngrams(n_gram: int, text: list) -> set:
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n_gram
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n_gram]))
    return ngram_set


def _block_trigrams(candidate: str, prediction: list) -> bool:
    tri_c = _get_ngrams(3, candidate.split())
    for s in prediction:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False


def _get_input_ids(
    tokenizer,
    src_txt,
    bert_compatible_cls=True,
    max_length=None,
):
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    if max_length is None:
        max_length = list(tokenizer.max_model_input_sizes.values())[0]
        if max_length > tokenizer.model_max_length:
            max_length = tokenizer.model_max_length

    if bert_compatible_cls:
        unk_token = str(tokenizer.unk_token)
        src_txt = [
            sent.replace(sep_token, unk_token).replace(cls_token, unk_token) for sent in src_txt
        ]

        if not len(src_txt) < 2:
            separation_string = " " + sep_token + " " + cls_token + " "
            text = separation_string.join(src_txt)
        else:
            try:
                text = src_txt[0]
            except IndexError:
                text = src_txt

        src_subtokens = tokenizer.tokenize(text)
        src_subtokens = src_subtokens[: (max_length - 2)]
        src_subtokens.insert(0, cls_token)
        src_subtokens.append(sep_token)
        input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
    else:
        input_ids = tokenizer.encode(
            src_txt,
            add_special_tokens=True,
            max_length=min(max_length, tokenizer.model_max_length),
        )

    return input_ids


def df_to_dataset(tokenizer, inputs=None, data_type="train"):
    logger.info("Processing %s data...", data_type)

    datasets = list()
    for idx, doc in inputs.iterrows():
        if idx % 1000 == 0:
            logger.info("Generating features for example %s/%s", idx, len(inputs))

        labels = np.zeros(len(doc["article"]), dtype=np.int64)
        labels[doc["extractive"]] = 1

        sources = [" ".join(sent) for sent in doc["article"]]
        input_ids = _get_input_ids(tokenizer, sources, bert_compatible_cls=True)
        attention_mask = [1] * len(input_ids)

        token_type_ids = []
        segment_flag = True
        for ids in input_ids:
            token_type_ids += [0 if segment_flag else 1]
            if ids == tokenizer.sep_token_id:
                segment_flag = not segment_flag

        sent_rep_id = tokenizer.sep_token_id
        sent_rep_token_ids = [i for i, t in enumerate(input_ids) if t == sent_rep_id]
        labels = labels[: len(sent_rep_token_ids)]

        datasets.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "sent_rep_token_ids": sent_rep_token_ids,
                "labels": labels,
                "sources": sources,
                "targets": doc["abstractive"],
            }
        )

    return np.array(datasets)


def preprocess_datasets(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, use_fast=True)
    print(tokenizer.model_max_length)
    print(tokenizer.sep_token)
    print(tokenizer.cls_token)
    print(tokenizer.unk_token)
    print(tokenizer.pad_token)
    print(tokenizer.sep_token_id)
    print(tokenizer.cls_token_id)
    print(tokenizer.unk_token_id)
    print(tokenizer.pad_token_id)

    datasets = dict()
    data_types = ["train", "valid"]
    for data_type in data_types:
        df = pd.read_pickle(f"{hparams.data_dir}/{data_type}_df.pkl")
        datasets[data_type] = df_to_dataset(tokenizer, df, data_type=data_type)
        np.save(
            os.path.join(hparams.data_dir, "dataset_" + data_type + "_small.npy"),
            datasets[data_type],
        )


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
        self.data_dir = hparams.data_dir
        self.batch_size = hparams.batch_size
        self.max_seq_len = hparams.max_seq_len
        self.datasets = dict()

    def prepare_data(self):
        data_types = ["train", "valid"]
        for data_type in data_types:
            self.datasets[data_type] = np.load(
                os.path.join(self.data_dir, "dataset_" + data_type + "_small.npy"),
                allow_pickle=True,
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            logger.info("Loading train data...")
        if stage == "test" or stage is None:
            logger.info("Loading test data...")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["valid"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
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


class KobertSummarization(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        self.model = AutoModel.from_pretrained("monologg/kobert")
        self.classifier = SimpleLinearClassifier(self.model.config.hidden_size)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.metric = load_metric("squad")

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
        data_dir="../data/ai.hub",
        model_dir="../data/ai.hub/checkpoints",
        model_file="../data/ai.hub/checkpoints/kobert-ext-sum.ckpt",
        load_from_checkpoint=False,
        model_name="monologg/kobert",
        learning_rate=1e-5,
        batch_size=32,
        num_epochs=100,
        max_seq_len=512,
        weight_decay=0.01,
        top_k_sentences=2,
    )

    # preprocess_datasets(hparams)

    dm = DataModule(hparams)

    if hparams.load_from_checkpoint:
        model = KobertSummarization.load_from_checkpoint(hparams.model_file, strict=False)
    else:
        model = KobertSummarization(hparams=hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.model_dir,
        filename="my-kobert-base-{epoch}",
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

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    # cnn_dm.prepare_data()
    # cnn_dm.setup(stage="test")
    #
    # idx = np.random.randint(0, 1000)
    # input_sentences = cnn_dm.datasets["test"][idx]["sources"]
    # predictions = model.predict_sentences(input_sentences, top_k=hparams.top_k_sentences)
    # print(predictions)
    # print(cnn_dm.datasets["test"][idx]["targets"])
