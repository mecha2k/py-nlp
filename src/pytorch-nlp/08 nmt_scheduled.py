import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import json
import re
import string

from argparse import Namespace
from collections import Counter
from nltk.translate import bleu_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

## 신경망 기계 번역 모델
# 1. NMTEncoder
#     - 소스 시퀀스를 입력으로 받아 임베딩하여 양방향 GRU에 주입합니다.
# 2. NMTDecoder
#     - 인코더 상태와 어텐션을 사용해 디코더가 새로운 시퀀스를 생성합니다.
#     - 타임 스텝마다 정답 타깃 시퀀스를 입력으로 사용합니다.
#     - 또는 디코더가 선택한 시퀀스를 입력으로 사용할 수도 있습니다.
#     - 이를 커리큘럼 학습(curriculum learning), 탐색 학습(learning to search)이라 부릅니다.
# 3. NMTModel
#     - 인코더와 디코더를 하나의 클래스로 구성합니다.
#
# 만약 코랩에서 실행하는 경우 아래 코드를 실행하여 전처리된 데이터를 다운로드하세요.
# get_ipython().system("mkdir data")
# get_ipython().system("wget https://git.io/JqQBE -O data/download.py")
# get_ipython().system("wget https://git.io/JqQB7 -O data/get-all-data.sh")
# get_ipython().system("chmod 755 data/get-all-data.sh")
# get_ipython().run_line_magic("cd", "data")
# get_ipython().system("./get-all-data.sh")
# get_ipython().run_line_magic("cd", "..")


class Vocabulary:
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {"token_to_idx": self._token_to_idx}

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


class NMTVectorizer:
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @staticmethod
    def _vectorize(indices, vector_length=-1, mask_index=0):
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[: len(indices)] = indices
        vector[len(indices) :] = mask_index
        return vector

    def _get_source_indices(self, text):
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text):
        indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(
            source_indices,
            vector_length=source_vector_length,
            mask_index=self.source_vocab.mask_index,
        )

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(
            target_x_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index,
        )
        target_y_vector = self._vectorize(
            target_y_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index,
        )

        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices),
        }

    @classmethod
    def from_dataframe(cls, text_df):
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in text_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            max_source_length=contents["max_source_length"],
            max_target_length=contents["max_target_length"],
        )

    def to_serializable(self):
        return {
            "source_vocab": self.source_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
        }


class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df.split == "test"]
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
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split == "train"]
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

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
        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {
            "x_source": vector_dict["source_vector"],
            "x_target": vector_dict["target_x_vector"],
            "y_target": vector_dict["target_y_vector"],
            "x_source_length": vector_dict["source_length"],
        }

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_nmt_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        lengths = data_dict["x_source_length"].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        super(NMTEncoder, self).__init__()

        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_source, x_lengths):
        x_embedded = self.source_embedding(x_source)
        # PackedSequence 생성; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(
            x_embedded, x_lengths.detach().cpu().numpy(), batch_first=True
        )

        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out, x_birnn_h = self.birnn(x_packed)
        # (batch_size, num_rnn, feature_size)로 변환
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        # 특성 펼침; (batch_size, num_rnn * feature_size)로 바꾸기
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h


def verbose_attention(encoder_state_vectors, query_vector):
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(
        encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), dim=2
    )
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores


def terse_attention(encoder_state_vectors, query_vector):
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(
        encoder_state_vectors.transpose(-2, -1), vector_probabilities.unsqueeze(dim=2)
    ).squeeze()
    return context_vectors, vector_probabilities


class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index):
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=0
        )
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size, rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.bos_index = bos_index
        self._sampling_temperature = 3

        self._cached_ht = None
        self._cached_p_attn = None
        self._cached_decoder_state = None

    def _init_indices(self, batch_size):
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index

    def _init_context_vectors(self, batch_size):
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
        if target_sequence is None:
            sample_probability = 1.0
            output_sequence_size = 0
        else:
            # 가정: 첫 번째 차원은 배치 차원입니다
            # 즉 입력은 (Batch, Seq) 시퀀스에 대해 반복해야 하므로 (Seq, Batch)로 차원을 바꿉니다
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        # 주어진 인코더의 은닉 상태를 초기 은닉 상태로 사용합니다
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # 문맥 벡터를 0으로 초기화합니다
        context_vectors = self._init_context_vectors(batch_size)
        # 첫 단어 y_t를 BOS로 초기화합니다
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            # 스케줄링된 샘플링 사용 여부
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            # 단계 1: 단어를 임베딩하고 이전 문맥과 연결합니다
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # 단계 2: GRU를 적용하고 새로운 은닉 벡터를 얻습니다
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())

            # 단계 3: 현재 은닉 상태를 사용해 인코더의 상태를 주목합니다
            context_vectors, p_attn, _ = verbose_attention(
                encoder_state_vectors=encoder_state, query_vector=h_t
            )

            # 부가 작업: 시각화를 위해 어텐션 확률을 저장합니다
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            # 단게 4: 현재 은닉 상태와 문맥 벡터를 사용해 다음 단어를 예측합니다
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                # _, y_t_index = torch.max(p_y_t_index, 1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            # 부가 작업: 예측 성능 점수를 기록합니다
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors


class NMTModel(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        source_embedding_size,
        target_vocab_size,
        target_embedding_size,
        encoding_size,
        target_bos_index,
    ):
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(
            num_embeddings=source_vocab_size,
            embedding_size=source_embedding_size,
            rnn_hidden_size=encoding_size,
        )
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(
            num_embeddings=target_vocab_size,
            embedding_size=target_embedding_size,
            rnn_hidden_size=decoding_size,
            bos_index=target_bos_index,
        )

    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability=0.0):
        # 매개변수:
        #     x_source (torch.Tensor): 소스 텍스트 데이터 텐서
        #         x_source.shape는 (batch, vectorizer.max_source_length)입니다.
        #     x_source_lengths torch.Tensor): x_source에 있는 시퀀스 길이
        #     target_sequence (torch.Tensor): 타깃 텍스트 데이터 텐서
        #     sample_probability (float): 스케줄링된 샘플링 파라미터
        #         디코더 타임 스텝마다 모델 예측에 사용할 확률
        # 반환값:
        #     decoded_states (torch.Tensor): 각 출력 타임 스텝의 예측 벡터
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(
            encoder_state=encoder_state,
            initial_hidden_state=final_hidden_states,
            target_sequence=target_sequence,
            sample_probability=sample_probability,
        )
        return decoded_states


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


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
    # 적어도 한 번 모델을 저장합니다
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state["epoch_index"] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]

        # 손실이 나빠지면
        if loss_t >= loss_tm1:
            # 조기 종료 단계 업데이트
            train_state["early_stopping_step"] += 1
        # 손실이 감소하면
        else:
            # 최상의 모델 저장
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
                train_state["early_stopping_best_val"] = loss_t

            # 조기 종료 단계 재설정
            train_state["early_stopping_step"] = 0

        # 조기 종료 여부 확인
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= args.early_stopping_criteria
        )

    return train_state


def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def sentence_from_indices(indices, vocab, strict=True, return_string=True):
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    if return_string:
        return " ".join(out)
    else:
        return out


class NMTSampler:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self._last_batch = None

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        y_pred = self.model(
            x_source=batch_dict["x_source"],
            x_source_lengths=batch_dict["x_source_length"],
            target_sequence=batch_dict["x_target"],
        )
        self._last_batch["y_pred"] = y_pred

        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch["attention"] = attention_batched

    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch["x_source"][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch["y_target"][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch["y_pred"], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index, return_string=True):
        output = {
            "source": self._get_source_sentence(index, return_string=return_string),
            "reference": self._get_reference_sentence(index, return_string=return_string),
            "sampled": self._get_sampled_sentence(index, return_string=return_string),
            "attention": self._last_batch["attention"][index],
        }

        reference = output["reference"]
        hypothesis = output["sampled"]

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)

        output["bleu-4"] = bleu_score.sentence_bleu(
            references=[reference], hypothesis=hypothesis, smoothing_function=chencherry.method1
        )

        return output


def get_source_sentence(vectorizer, batch_dict, index):
    indices = batch_dict["x_source"][index].cpu().data.numpy()
    vocab = vectorizer.source_vocab
    return sentence_from_indices(indices, vocab)


def get_true_sentence(vectorizer, batch_dict, index):
    return sentence_from_indices(
        batch_dict["y_target"].cpu().data.numpy()[index], vectorizer.target_vocab
    )


def get_sampled_sentence(vectorizer, batch_dict, index):
    y_pred = model(
        x_source=batch_dict["x_source"],
        x_source_lengths=batch_dict["x_source_length"],
        target_sequence=batch_dict["x_target"],
        sample_probability=1.0,
    )
    return sentence_from_indices(
        torch.max(y_pred, dim=2)[1].cpu().data.numpy()[index], vectorizer.target_vocab
    )


def get_all_sentences(vectorizer, batch_dict, index):
    return {
        "source": get_source_sentence(vectorizer, batch_dict, index),
        "truth": get_true_sentence(vectorizer, batch_dict, index),
        "sampled": get_sampled_sentence(vectorizer, batch_dict, index),
    }


def sentence_from_indices(indices, vocab, strict=True):
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.lookup_index(index))
    return " ".join(out)


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="../data/nmt/simplest_eng_fra.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="../data/nmt/models",
        reload_from_files=False,
        expand_filepath_to_save_dir=True,
        cuda=True,
        seed=42,
        learning_rate=5e-4,
        batch_size=256,
        num_epochs=1,
        early_stopping_criteria=5,
        source_embedding_size=24,
        target_embedding_size=24,
        encoding_size=32,
        catch_keyboard_interrupt=True,
    )

    if args.expand_filepath_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("CUDA 사용 여부: {}".format(args.cuda))

    handle_dirs(args.save_dir)
    set_seed_everywhere(args.seed, args.cuda)

    if args.reload_from_files and os.path.exists(args.vectorizer_file):
        dataset = NMTDataset.load_dataset_and_load_vectorizer(
            args.dataset_csv, args.vectorizer_file
        )
    else:
        dataset = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    model = NMTModel(
        source_vocab_size=len(vectorizer.source_vocab),
        source_embedding_size=args.source_embedding_size,
        target_vocab_size=len(vectorizer.target_vocab),
        target_embedding_size=args.target_embedding_size,
        encoding_size=args.encoding_size,
        target_bos_index=vectorizer.target_vocab.begin_seq_index,
    )

    if args.reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        print("로드한 모델")
    else:
        print("새로운 모델")

    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=1
    )
    mask_index = vectorizer.target_vocab.mask_index
    train_state = make_train_state(args)

    epoch_bar = tqdm(desc="training routine", total=args.num_epochs, position=0)

    dataset.set_split("train")
    train_bar = tqdm(
        desc="split=train", total=dataset.get_num_batches(args.batch_size), position=1, leave=True
    )
    dataset.set_split("val")
    val_bar = tqdm(
        desc="split=val", total=dataset.get_num_batches(args.batch_size), position=1, leave=True
    )

    try:
        for epoch_index in range(args.num_epochs):
            sample_probability = (20 + epoch_index) / args.num_epochs
            train_state["epoch_index"] = epoch_index

            dataset.set_split("train")
            batch_generator = generate_nmt_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()

                y_pred = model(
                    batch_dict["x_source"],
                    batch_dict["x_source_length"],
                    batch_dict["x_target"],
                    sample_probability=sample_probability,
                )

                loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)
                loss.backward()
                optimizer.step()

                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # 진행 상태 막대 업데이트
                train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                train_bar.update()

            train_state["train_loss"].append(running_loss)
            train_state["train_acc"].append(running_acc)

            dataset.set_split("val")
            batch_generator = generate_nmt_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = model(
                    batch_dict["x_source"],
                    batch_dict["x_source_length"],
                    batch_dict["x_target"],
                    sample_probability=sample_probability,
                )

                loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                val_bar.update()

            train_state["val_loss"].append(running_loss)
            train_state["val_acc"].append(running_acc)

            train_state = update_train_state(args=args, model=model, train_state=train_state)
            scheduler.step(train_state["val_loss"][-1])

            if train_state["stop_early"]:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state["early_stopping_best_val"])
            epoch_bar.update()

    except KeyboardInterrupt:
        print("반복 중지")

    chencherry = bleu_score.SmoothingFunction()

    model = model.eval().to(args.device)
    sampler = NMTSampler(vectorizer, model)

    dataset.set_split("test")
    batch_generator = generate_nmt_batches(dataset, batch_size=args.batch_size, device=args.device)

    test_results = []
    for batch_dict in batch_generator:
        sampler.apply_to_batch(batch_dict)
        for i in range(args.batch_size):
            test_results.append(sampler.get_ith_item(i, False))

    plt.hist([r["bleu-4"] for r in test_results], bins=100)
    plt.savefig("bleu_hist.png", dpi=200)
    print("bleu-4 mean: ", np.mean([r["bleu-4"] for r in test_results]))
    print("bleu-4 median: ", np.median([r["bleu-4"] for r in test_results]))

    dataset.set_split("val")
    batch_generator = generate_nmt_batches(dataset, batch_size=args.batch_size, device=args.device)
    batch_dict = next(batch_generator)

    model = model.eval().to(args.device)
    sampler = NMTSampler(vectorizer, model)
    sampler.apply_to_batch(batch_dict)

    all_results = []
    for i in range(args.batch_size):
        all_results.append(sampler.get_ith_item(i, False))

    top_results = [x for x in all_results if x["bleu-4"] > 0.5]
    len(top_results)

    for sample in top_results:
        plt.figure(figsize=(6, 4))
        target_len = len(sample["sampled"])
        source_len = len(sample["source"])

        attention_matrix = sample["attention"][:target_len, : source_len + 2].transpose()  # [::-1]
        ax = sns.heatmap(attention_matrix, center=0.0)
        ylabs = ["<BOS>"] + sample["source"] + ["<EOS>"]
        ax.set_yticklabels(ylabs, rotation=0)
        ax.set_xticklabels(sample["sampled"], rotation=90)
        ax.set_xlabel("Target Sentence")
        ax.set_ylabel("Source Sentence\n\n")
        ax.set_title("Attention Matrix")
        plt.tight_layout()
        print(f"{args.save_dir}/{sample['id']}")
        plt.savefig(f"{args.save_dir}/{sample['id']}", dpi=200)

    results = get_all_sentences(vectorizer, batch_dict, 1)
    print(results)
