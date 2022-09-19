import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from transformers import utils
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser, Namespace
from glob import glob
import logging
import os


class ChatDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):
        self._df = df
        self.que = usr_token
        self.ans = sys_token
        self.eos = eos_token
        self.pad = pad_token
        self.sent = sent_token
        self.mask = mask_token
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        df = self._df.iloc[idx]
        que = df.user
        ans = df.system
        sentiment = df.sentiment
        que_tokens = self.tokenizer.tokenize(self.que + que)
        ans_tokens = self.tokenizer.tokenize(self.sent + sentiment + self.ans + ans + self.eos)
        que_len = len(que_tokens)
        ans_len = len(ans_tokens)
        if que_len + ans_len > self.max_len:
            ans_len = self.max_len - que_len
            if ans_len <= 0:
                que_tokens = que_tokens[-(int(self.max_len / 2)) :]
                que_len = len(que_tokens)
                ans_len = self.max_len - que_len
                assert ans_len > 0
            ans_tokens = ans_tokens[:ans_len]
            ans_len = len(ans_tokens)
            assert ans_len == len(ans_tokens), f"{ans_len} ==? {len(ans_tokens)}"
        labels = [
            self.mask,
        ] * que_len + ans_tokens[1:]
        masks = [0] * que_len + [1] * ans_len + [0] * (self.max_len - que_len - ans_len)
        label_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(label_ids) < self.max_len:
            label_ids += [self.tokenizer.pad_token_id]
        input_ids = self.tokenizer.convert_tokens_to_ids(que_tokens + ans_tokens)
        while len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "masks": torch.tensor(masks, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }


def _collate_fn(batch):
    input_ids, masks, label_ids = zip(*batch)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, masks, label_ids


class KoGPT2Model(LightningModule):
    def __init__(self, tokenizer, params):
        super().__init__()
        self.save_hyperparameters()
        self._tokenizer = tokenizer
        self._max_len = params.max_len
        self._epochs = params.epochs
        self._negative = params.negative
        self._learning_rate = params.learning_rate
        self._warmup_ratio = params.warmup_ratio
        self._batch_size = params.batch_size
        self._source_file = params.source_file
        self._model_file = params.model_file
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        if os.path.exists(params.model_file):
            self.kogpt2 = GPT2LMHeadModel.from_pretrained(params.model_file)
        else:
            self.kogpt2 = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def forward(self, input_ids):
        outputs = self.kogpt2(input_ids=input_ids, return_dict=True)
        return outputs.logits

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self._learning_rate)
        num_training_steps = self._epochs * len(self.train_dataloader())
        num_warmup_steps = int(num_training_steps * self._warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "cosine_schedule_with_warmup",
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, masks, labels = batch.values()
        outputs = self(inputs)
        masks_3d = masks.unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
        masks_out = torch.where(masks_3d == 1, outputs, self._negative * torch.ones_like(outputs))
        loss = self.loss_fn(masks_out.transpose(1, 2), labels)
        loss = loss.sum() / batch["masks"].sum()
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        df = pd.read_csv(self._source_file)
        df.drop_duplicates(subset=["user", "system"], inplace=True)
        df.dropna(subset=["user", "system"], how="any", inplace=True)
        train_dataset = ChatDataset(df, self._tokenizer, self._max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        return train_dataloader


if __name__ == "__main__":
    utils.logging.set_verbosity_error()
    parser = ArgumentParser(description="KoGPT-2 chatbot")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available in torch")

    unk_token = "<unk>"
    usr_token = "<usr>"
    sys_token = "<sys>"
    bos_token = "<s>"
    eos_token = "</s>"
    mask_token = "<mask>"
    pad_token = "<pad>"
    sent_token = "<unused0>"

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    print("vocab size : ", len(tokenizer.get_vocab()))
    # tokenizer.add_special_tokens({"sent_token": sent_token})
    # assert tokenizer.sent_token == "<unused0>"
    # print("vocab size : ", len(tokenizer.get_vocab()))
    print("all_special_tokens : ", tokenizer.all_special_tokens)
    print("tokens (10) : ", [tokenizer.decode(i) for i in range(10)])

    checkpoint_callback = ModelCheckpoint(
        dirpath="../data/gpt-2/py-models",
        filename="{epoch:02d}-{train_loss:.2f}",
        verbose=True,
        save_last=True,
        monitor="train_loss",
        mode="min",
    )

    params = Namespace(
        max_len=32,
        epochs=1,
        batch_size=64,
        negative=-1e18,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        source_file="../data/gpt-2/chatbot_dataset.csv",
        model_file="../data/gpt-2/py-models/kogpt2-chatbot-v1",
    )
    model = KoGPT2Model(tokenizer, params)
    model.train()

    logger = loggers.TensorBoardLogger(save_dir="../data/gpt-2/py-models")
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=params.epochs,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model)

    trained_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trained_model.kogpt2.save_pretrained(params.model_file)

    for file in glob("../data/gpt-2/py-models/*.ckpt"):
        os.remove(file)
