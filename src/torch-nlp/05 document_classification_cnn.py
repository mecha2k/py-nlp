import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


class SequenceVocabulary(Vocabulary):
    def __init__(
        self,
        token_to_idx=None,
        unk_token="<UNK>",
        mask_token="<MASK>",
        begin_seq_token="<BEGIN>",
        end_seq_token="<END>",
    ):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update(
            {
                "unk_token": self._unk_token,
                "mask_token": self._mask_token,
                "begin_seq_token": self._begin_seq_token,
                "end_seq_token": self._end_seq_token,
            }
        )
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class NewsVectorizer:
    def __init__(self, title_vocab, category_vocab):
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab

    def vectorize(self, title, vector_length=-1):
        indices = [self.title_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token) for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)
        if vector_length < 0:
            vector_length = len(indices)
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[: len(indices)] = indices
        out_vector[len(indices) :] = self.title_vocab.mask_index
        return out_vector

    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        category_vocab = Vocabulary()
        for category in sorted(set(news_df.category)):
            category_vocab.add_token(category)

        word_counts = Counter()
        for title in news_df.title:
            for token in title.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        title_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)

        return cls(title_vocab, category_vocab)

    @classmethod
    def from_serializable(cls, contents):
        title_vocab = SequenceVocabulary.from_serializable(contents["title_vocab"])
        category_vocab = Vocabulary.from_serializable(contents["category_vocab"])

        return cls(title_vocab=title_vocab, category_vocab=category_vocab)

    def to_serializable(self):
        return {
            "title_vocab": self.title_vocab.to_serializable(),
            "category_vocab": self.category_vocab.to_serializable(),
        }


class NewsDataset(Dataset):
    def __init__(self, news_df, vectorizer):
        self.news_df = news_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, news_df.title)) + 2

        self.train_df = self.news_df[self.news_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.news_df[self.news_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.news_df[self.news_df.split == "test"]
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
        class_counts = news_df.category.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.category_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv):
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split == "train"]
        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, news_csv, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(news_csv, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NameVectorizer.from_serializable(json.load(fp))

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
        title_vector = self._vectorizer.vectorize(row.title, self._max_seq_length)
        category_index = self._vectorizer.category_vocab.lookup_token(row.category)
        return {"x_data": title_vector, "y_target": category_index}

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


class NewsClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_embeddings,
        num_channels,
        hidden_dim,
        num_classes,
        dropout_p,
        pretrained_embeddings=None,
        padding_idx=0,
    ):
        super(NewsClassifier, self).__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(
                embedding_dim=embedding_size, num_embeddings=num_embeddings, padding_idx=padding_idx
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
        )

        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        # 평균 값을 계산하여 부가적인 차원을 제거합니다
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)

        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)

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


def load_glove_from_file(glove_filepath):
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    final_embeddings = np.zeros((len(words), embedding_size))
    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i
    return final_embeddings


if __name__ == "__main__":
    args = Namespace(
        news_csv="../data/ag_news/news_with_splits.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="../data/ag_news",
        glove_filepath="../data/glove/glove.6B.100d.txt",
        use_glove=True,
        embedding_size=100,
        hidden_dim=100,
        num_channels=100,
        seed=42,
        learning_rate=0.001,
        dropout_p=0.1,
        batch_size=128,
        num_epochs=10,
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
        dataset = NewsDataset.load_dataset_and_load_vectorizer(args.news_csv, args.vectorizer_file)
    else:
        dataset = NewsDataset.load_dataset_and_make_vectorizer(args.news_csv)
        dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    if args.use_glove:
        words = vectorizer.title_vocab._token_to_idx.keys()
        embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath, words=words)
        print("사전 훈련된 임베딩을 사용합니다")
    else:
        print("사전 훈련된 임베딩을 사용하지 않습니다")
        embeddings = None

    classifier = NewsClassifier(
        embedding_size=args.embedding_size,
        num_embeddings=len(vectorizer.title_vocab),
        num_channels=args.num_channels,
        hidden_dim=args.hidden_dim,
        num_classes=len(vectorizer.category_vocab),
        dropout_p=args.dropout_p,
        pretrained_embeddings=embeddings,
        padding_idx=0,
    )

    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)

    loss_func = nn.CrossEntropyLoss(dataset.class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=1
    )

    train_state = make_train_state(args)
    epoch_bar = tqdm(desc="training routine", total=args.num_epochs, position=0)

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
                y_pred = classifier(batch_dict["x_data"])
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
                y_pred = classifier(batch_dict["x_data"])
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
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    dataset.set_split("test")
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        y_pred = classifier(batch_dict["x_data"])
        loss = loss_func(y_pred, batch_dict["y_target"])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state["test_loss"] = running_loss
    train_state["test_acc"] = running_acc

    print("테스트 손실: {};".format(train_state["test_loss"]))
    print("테스트 정확도: {}".format(train_state["test_acc"]))

    def preprocess_text(text):
        text = " ".join(word.lower() for word in text.split(" "))
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        return text

    def predict_category(title, classifier, vectorizer, max_length):
        title = preprocess_text(title)
        vectorized_title = torch.tensor(vectorizer.vectorize(title, vector_length=max_length))
        result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
        probability_values, indices = result.max(dim=1)
        predicted_category = vectorizer.category_vocab.lookup_index(indices.item())
        return {"category": predicted_category, "probability": probability_values.item()}

    def get_samples():
        samples = {}
        for cat in dataset.val_df.category.unique():
            samples[cat] = dataset.val_df.title[dataset.val_df.category == cat].tolist()[:5]
        return samples

    val_samples = get_samples()
    classifier = classifier.to("cpu")

    for truth, sample_group in val_samples.items():
        print(f"True Category: {truth}")
        print("=" * 30)
        for sample in sample_group:
            prediction = predict_category(
                sample, classifier, vectorizer, dataset._max_seq_length + 1
            )
            print("예측: {} (p={:0.2f})".format(prediction["category"], prediction["probability"]))
            print("\t + 샘플: {}".format(sample))
        print("-" * 30 + "\n")
