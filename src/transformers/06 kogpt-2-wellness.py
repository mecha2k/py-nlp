import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

parser = argparse.ArgumentParser(description="MentalHealth-bot based on KoGPT-2")

parser.add_argument(
    "--model_params",
    type=str,
    default="model_chp/model_-last.ckpt",
    help="model binary for starting chat",
)

parser.add_argument("--train", action="store_true", default=False, help="for training")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = "<usr>"
S_TKN = "<sys>"
EOS = "</s>"
MASK = "<unused0>"
PAD = "<pad>"
SENT = "<unused1>"

TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK
)


max_len = 32
epochs = 3
batch_size = 64
warmup_ratio = 0.1


class CharDataset(Dataset):
    def __init__(self, chats, max_len=361):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.sent_token = SENT
        self.max_len = max_len
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        sent = turn["sentiment"]
        q = turn["user"]
        a = turn["system"]
        q_toked = self.tokenizer.tokenize(self.q_token + q)
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.sent_token + sent + self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)) :]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f"{a_len} ==? {len(a_toked)}"
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info("labels {}".format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return token_ids, np.array(mask), labels_ids


class KoGPT2Chat(LightningModule):
    def __init__(self, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.train_set = None
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max-len", type=int, default=32, help="max sentence length on input (default: 32)"
        )

        parser.add_argument(
            "--batch-size", type=int, default=96, help="batch size for training (default: 96)"
        )
        parser.add_argument("--lr", type=float, default=5e-5, help="The initial learning rate")
        parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio")
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log("train_loss", loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * epochs
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "cosine_schedule_with_warmup",
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def _collate_fn(batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        print(os.getcwd())
        data = pd.read_csv("../data/gpt-2/chatbot_dataset.csv")
        self.train_set = CharDataset(data, max_len=max_len)
        train_dataloader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        return train_dataloader


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

file = "../data/gpt-2/py-models"

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath="../data/gpt-2/py-models",
            filename="{epoch:02d}-{train_loss:.2f}",
            verbose=True,
            save_last=True,
            monitor="train_loss",
            mode="min",
        )
        # python train_torch.py --train --gpus 3 --max_epochs 3
        model = KoGPT2Chat()
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            accelerator="cpu",
            callbacks=checkpoint_callback,
            gradient_clip_val=1.0,
        )
        trainer.fit(model)
        # model.kogpt2.save_pretrained('./finetuned_model/')

        logging.info("best model path {}".format(checkpoint_callback.best_model_path))
        trained_model = model.kogpt2.load_from_checkpoint(checkpoint_callback.best_model_path)
        trained_model.save_pretrained("./finetuned_model/")