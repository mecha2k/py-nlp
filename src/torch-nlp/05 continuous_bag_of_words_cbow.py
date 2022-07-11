import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import json
import re
import tqdm
import string
from argparse import Namespace
from collections import Counter


class Vocabulary:
    def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
            "mask_token": self._mask_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class CBOWVectorizer:
    def __init__(self, cbow_vocab):
        self.cbow_vocab = cbow_vocab

    def vectorize(self, context, vector_length=-1):
        indices = [self.cbow_vocab.lookup_token(token) for token in context.split(" ")]
        if vector_length < 0:
            vector_length = len(indices)
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[: len(indices)] = indices
        out_vector[len(indices) :] = self.cbow_vocab.mask_index
        return out_vector

    @classmethod
    def from_dataframe(cls, cbow_df):
        cbow_vocab = Vocabulary()
        for index, row in cbow_df.iterrows():
            for token in row.context.split(" "):
                cbow_vocab.add_token(token)
            cbow_vocab.add_token(row.target)
        return cls(cbow_vocab)

    @classmethod
    def from_serializable(cls, contents):
        cbow_vocab = Vocabulary.from_serializable(contents["cbow_vocab"])
        return cls(cbow_vocab=cbow_vocab)

    def to_serializable(self):
        return {"cbow_vocab": self.cbow_vocab.to_serializable()}


class CBOWDataset(Dataset):
    def __init__(self, cbow_df, vectorizer):
        self.cbow_df = cbow_df
        self._vectorizer = vectorizer

        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, cbow_df.context))

        self.train_df = self.cbow_df[self.cbow_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.cbow_df[self.cbow_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.cbow_df[self.cbow_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        self._target_df = None
        self._target_size = None
        self._target_split = None
        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(cls, cbow_csv):
        cbow_df = pd.read_csv(cbow_csv)
        train_cbow_df = cbow_df[cbow_df.split == "train"]
        return cls(cbow_df, CBOWVectorizer.from_dataframe(train_cbow_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, cbow_csv, vectorizer_filepath):
        cbow_df = pd.read_csv(cbow_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(cbow_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return CBOWVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        context_vector = self._vectorizer.vectorize(row.context, self._max_seq_length)
        target_index = self._vectorizer.cbow_vocab.lookup_token(row.target)
        return {"x_data": context_vector, "y_target": target_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class CBOWClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        super(CBOWClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=vocabulary_size)

    def forward(self, x_in, apply_softmax=False):
        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        return y_out


def make_train_state(args):
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": args.learning_rate,
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": args.model_state_file,
    }


def update_train_state(args, model, train_state):
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False
    elif train_state["epoch_index"] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]
        if loss_t >= train_state["early_stopping_best_val"]:
            train_state["early_stopping_step"] += 1
        else:
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
            train_state["early_stopping_step"] = 0
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= args.early_stopping_criteria
        )
    return train_state


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


if __name__ == "__main__":
    args = Namespace(
        cbow_csv="../data/books/frankenstein_with_splits.csv",
        vectorizer_file="vectorizer_cbow.json",
        model_state_file="model_cbow.pth",
        save_dir="../data/books",
        embedding_size=50,
        seed=42,
        num_epochs=10,
        learning_rate=0.0001,
        batch_size=32,
        early_stopping_criteria=5,
        cuda=True,
        catch_keyboard_interrupt=True,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True,
    )

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("CUDA 사용 여부: {}".format(args.cuda))
    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)

    if args.reload_from_files:
        print("데이터셋과 Vectorizer를 로드합니다")
        dataset = CBOWDataset.load_dataset_and_load_vectorizer(args.cbow_csv, args.vectorizer_file)
    else:
        print("데이터셋을 로드하고 Vectorizer를 만듭니다")
        dataset = CBOWDataset.load_dataset_and_make_vectorizer(args.cbow_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    classifier = CBOWClassifier(
        vocabulary_size=len(vectorizer.cbow_vocab), embedding_size=args.embedding_size
    )

    classifier = classifier.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=1
    )
    train_state = make_train_state(args)
    epoch_bar = tqdm.tqdm(desc="training routine", total=args.num_epochs, position=0)

    try:
        for epoch_index in range(args.num_epochs):
            train_state["epoch_index"] = epoch_index
            dataset.set_split("train")
            batch_generator = generate_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()
                y_pred = classifier(x_in=batch_dict["x_data"])
                loss = loss_func(y_pred, batch_dict["y_target"])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                loss.backward()
                optimizer.step()

                acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state["train_loss"].append(running_loss)
            train_state["train_acc"].append(running_acc)

            dataset.set_split("val")
            batch_generator = generate_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = classifier(x_in=batch_dict["x_data"])
                loss = loss_func(y_pred, batch_dict["y_target"])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state["val_loss"].append(running_loss)
            train_state["val_acc"].append(running_acc)
            train_state = update_train_state(args=args, model=classifier, train_state=train_state)
            scheduler.step(train_state["val_loss"][-1])

            epoch_bar.update()
            if train_state["stop_early"]:
                break

    except KeyboardInterrupt:
        print("Exiting loop")

    classifier.load_state_dict(torch.load(train_state["model_filename"]))
    classifier = classifier.to(args.device)
    loss_func = nn.CrossEntropyLoss()

    dataset.set_split("test")
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        y_pred = classifier(x_in=batch_dict["x_data"])
        loss = loss_func(y_pred, batch_dict["y_target"])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state["test_loss"] = running_loss
    train_state["test_acc"] = running_acc
    print("테스트 손실: {};".format(train_state["test_loss"]))
    print("테스트 정확도: {}".format(train_state["test_acc"]))

    def pretty_print(results):
        for item in results:
            print("...[%.2f] - %s" % (item[1], item[0]))

    def get_closest(target_word, word_to_idx, embeddings, n=5):
        word_embedding = embeddings[word_to_idx[target_word.lower()]]
        distances = []
        for word, index in word_to_idx.items():
            if word == "<MASK>" or word == target_word:
                continue
            distances.append((word, torch.dist(word_embedding, embeddings[index])))
        results = sorted(distances, key=lambda x: x[1])[1 : n + 2]
        return results

    word = input("단어를 입력해 주세요: ")
    embeddings = classifier.embedding.weight.data
    word_to_idx = vectorizer.cbow_vocab._token_to_idx
    pretty_print(get_closest(word, word_to_idx, embeddings, n=5))

    target_words = ["frankenstein", "monster", "science", "sickness", "lonely", "happy"]
    embeddings = classifier.embedding.weight.data
    word_to_idx = vectorizer.cbow_vocab._token_to_idx

    for target_word in target_words:
        print(f"======={target_word}=======")
        if target_word not in word_to_idx:
            print("Not in vocabulary")
            continue
        pretty_print(get_closest(target_word, word_to_idx, embeddings, n=5))
