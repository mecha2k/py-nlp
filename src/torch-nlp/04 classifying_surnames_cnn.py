from argparse import Namespace
from collections import Counter
import json
import os
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm


class Vocabulary:
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        try:
            index = self._token_to_idx[token]
        except KeyError:
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
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SurnameVectorizer:
    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    def vectorize(self, surname):
        one_hot_matrix_size = (len(self.surname_vocab), self._max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        for position_index, character in enumerate(surname):
            character_index = self.surname_vocab.lookup_token(character)
            one_hot_matrix[character_index][position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length)

    @classmethod
    def from_serializable(cls, contents):
        surname_vocab = Vocabulary.from_serializable(contents["surname_vocab"])
        nationality_vocab = Vocabulary.from_serializable(contents["nationality_vocab"])
        return cls(
            surname_vocab=surname_vocab,
            nationality_vocab=nationality_vocab,
            max_surname_length=contents["max_surname_length"],
        )

    def to_serializable(self):
        return {
            "surname_vocab": self.surname_vocab.to_serializable(),
            "nationality_vocab": self.nationality_vocab.to_serializable(),
            "max_surname_length": self._max_surname_length,
        }


class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer
        self.train_df = self.surname_df[self.surname_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split == "test"]
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

        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split == "train"]
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, surname_csv, vectorizer_filepath):
        surname_df = pd.read_csv(surname_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(surname_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return SurnameVectorizer.from_serializable(json.load(fp))

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
        surname_matrix = self._vectorizer.vectorize(row.surname)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {"x_surname": surname_matrix, "y_nationality": nationality_index}

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


class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SurnameClassifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_surname, apply_softmax=False):
        features = self.convnet(x_surname).squeeze(dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector


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
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


if __name__ == "__main__":
    args = Namespace(
        surname_csv="../data/surnames/surnames_with_splits_cnn.csv",
        vectorizer_file="vectorizer_cnn.json",
        model_state_file="model_cnn.pth",
        save_dir="../data/surnames",
        hidden_dim=100,
        num_channels=256,
        seed=42,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=10,
        early_stopping_criteria=5,
        dropout_p=0.1,
        cuda=True,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True,
        catch_keyboard_interrupt=True,
    )

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    def set_seed_everywhere(seed, cuda):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)

    def handle_dirs(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("CUDA 사용 여부: {}".format(args.cuda))
    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)

    if args.reload_from_files:
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(
            args.surname_csv, args.vectorizer_file
        )
    else:
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    classifier = SurnameClassifier(
        initial_num_channels=len(vectorizer.surname_vocab),
        num_classes=len(vectorizer.nationality_vocab),
        num_channels=args.num_channels,
    )

    classifer = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)

    loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights)
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
                y_pred = classifier(batch_dict["x_surname"])
                loss = loss_func(y_pred, batch_dict["y_nationality"])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                loss.backward()
                optimizer.step()
                acc_t = compute_accuracy(y_pred, batch_dict["y_nationality"])
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
                y_pred = classifier(batch_dict["x_surname"])
                loss = loss_func(y_pred, batch_dict["y_nationality"])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict["y_nationality"])
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
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    dataset.set_split("test")
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        y_pred = classifier(batch_dict["x_surname"])
        loss = loss_func(y_pred, batch_dict["y_nationality"])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["y_nationality"])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state["test_loss"] = running_loss
    train_state["test_acc"] = running_acc
    print("테스트 손실: {};".format(train_state["test_loss"]))
    print("테스트 정확도: {}".format(train_state["test_acc"]))

    def predict_nationality(surname, classifier, vectorizer):
        vectorized_surname = vectorizer.vectorize(surname)
        vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
        result = classifier(vectorized_surname, apply_softmax=True)
        probability_values, indices = result.max(dim=1)
        index = indices.item()
        predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
        probability_value = probability_values.item()
        return {"nationality": predicted_nationality, "probability": probability_value}

    new_surname = input("분류하려는 성씨를 입력하세요: ")
    classifier = classifier.cpu()
    prediction = predict_nationality(new_surname, classifier, vectorizer)
    print(f"{new_surname} -> {prediction['nationality']} (p={prediction['probability']:0.2f})")

    def predict_topk_nationality(surname, classifier, vectorizer, k=5):
        vectorized_surname = vectorizer.vectorize(surname)
        vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
        prediction_vector = classifier(vectorized_surname, apply_softmax=True)
        probability_values, indices = torch.topk(prediction_vector, k=k)
        probability_values = probability_values[0].detach().numpy()
        indices = indices[0].detach().numpy()

        results = []
        for kth_index in range(k):
            nationality = vectorizer.nationality_vocab.lookup_index(indices[kth_index])
            probability_value = probability_values[kth_index]
            results.append({"nationality": nationality, "probability": probability_value})
        return results

    new_surname = input("분류하려는 성씨를 입력하세요: ")
    k = int(input("얼마나 많은 예측을 보고 싶나요? "))
    if k > len(vectorizer.nationality_vocab):
        print("앗! 전체 국적 개수보다 큰 값을 입력했습니다. 모든 국적에 대한 예측을 반환합니다. :)")
        k = len(vectorizer.nationality_vocab)

    predictions = predict_topk_nationality(new_surname, classifier, vectorizer, k=k)
    print("최상위 {}개 예측:".format(k))
    print("===================")
    for prediction in predictions:
        print(f"{new_surname} -> {prediction['nationality']} (p={prediction['probability']:0.2f})")
