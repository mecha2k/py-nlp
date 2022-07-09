#!/usr/bin/env python
# coding: utf-8

# *아래 링크를 통해 이 노트북을 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있습니다.*
#
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.org/github/rickiepark/nlp-with-pytorch/blob/master/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb"><img src="https://jupyter.org/assets/share.png" width="60" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/nlp-with-pytorch/blob/master/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# # 레스토랑 리뷰 감성 분류하기

# ## 임포트

# In[1]:


from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm


# ## 데이터 벡터화 클래스

# ### `Dataset`

# In[2]:


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        매개변수:
            review_df (pandas.DataFrame): 데이터셋
            vectorizer (ReviewVectorizer): ReviewVectorizer 객체
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 만듭니다

        매개변수:
            review_csv (str): 데이터셋의 위치
        반환값:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        train_review_df = review_df[review_df.split == "train"]
        return cls(review_df, ReviewVectorizer.from_dataframe(train_review_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
        """데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 만듭니다.
        캐시된 ReviewVectorizer 객체를 재사용할 때 사용합니다.

        매개변수:
            review_csv (str): 데이터셋의 위치
            vectorizer_filepath (str): ReviewVectorizer 객체의 저장 위치
        반환값:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """파일에서 ReviewVectorizer 객체를 로드하는 정적 메서드

        매개변수:
            vectorizer_filepath (str): 직렬화된 ReviewVectorizer 객체의 위치
        반환값:
            ReviewVectorizer의 인스턴스
        """
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """ReviewVectorizer 객체를 json 형태로 디스크에 저장합니다

        매개변수:
            vectorizer_filepath (str): ReviewVectorizer 객체의 저장 위치
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """벡터 변환 객체를 반환합니다"""
        return self._vectorizer

    def set_split(self, split="train"):
        """데이터프레임에 있는 열을 사용해 분할 세트를 선택합니다

        매개변수:
            split (str): "train", "val", "test" 중 하나
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드

        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target)로 이루어진 딕셔너리
        """
        row = self._target_df.iloc[index]

        review_vector = self._vectorizer.vectorize(row.review)

        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {"x_data": review_vector, "y_target": rating_index}

    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다

        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size


# ### `Vocabulary`

# In[3]:


class Vocabulary(object):
    """매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
            add_unk (bool): UNK 토큰을 추가할지 지정하는 플래그
            unk_token (str): Vocabulary에 추가할 UNK 토큰
        """

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
        """직렬화할 수 있는 딕셔너리를 반환합니다"""
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        """직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다"""
        return cls(**contents)

    def add_token(self, token):
        """토큰을 기반으로 매핑 딕셔너리를 업데이트합니다

        매개변수:
            token (str): Vocabulary에 추가할 토큰
        반환값:
            index (int): 토큰에 상응하는 정수
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """토큰 리스트를 Vocabulary에 추가합니다.

        매개변수:
            tokens (list): 문자열 토큰 리스트
        반환값:
            indices (list): 토큰 리스트에 상응되는 인덱스 리스트
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """토큰에 대응하는 인덱스를 추출합니다.
        토큰이 없으면 UNK 인덱스를 반환합니다.

        매개변수:
            token (str): 찾을 토큰
        반환값:
            index (int): 토큰에 해당하는 인덱스
        노트:
            UNK 토큰을 사용하려면 (Vocabulary에 추가하기 위해)
            `unk_index`가 0보다 커야 합니다.
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """인덱스에 해당하는 토큰을 반환합니다.

        매개변수:
            index (int): 찾을 인덱스
        반환값:
            token (str): 인텍스에 해당하는 토큰
        에러:
            KeyError: 인덱스가 Vocabulary에 없을 때 발생합니다.
        """
        if index not in self._idx_to_token:
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


# ### `Vectorizer`

# In[4]:


class ReviewVectorizer(object):
    """어휘 사전을 생성하고 관리합니다"""

    def __init__(self, review_vocab, rating_vocab):
        """
        매개변수:
            review_vocab (Vocabulary): 단어를 정수에 매핑하는 Vocabulary
            rating_vocab (Vocabulary): 클래스 레이블을 정수에 매핑하는 Vocabulary
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """리뷰에 대한 웟-핫 벡터를 만듭니다

        매개변수:
            review (str): 리뷰
        반환값:
            one_hot (np.ndarray): 원-핫 벡터
        """
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """데이터셋 데이터프레임에서 Vectorizer 객체를 만듭니다

        매개변수:
            review_df (pandas.DataFrame): 리뷰 데이터셋
            cutoff (int): 빈도 기반 필터링 설정값
        반환값:
            ReviewVectorizer 객체
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # 점수를 추가합니다
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # count > cutoff인 단어를 추가합니다
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """직렬화된 딕셔너리에서 ReviewVectorizer 객체를 만듭니다

        매개변수:
            contents (dict): 직렬화된 딕셔너리
        반환값:
            ReviewVectorizer 클래스 객체
        """
        review_vocab = Vocabulary.from_serializable(contents["review_vocab"])
        rating_vocab = Vocabulary.from_serializable(contents["rating_vocab"])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """캐싱을 위해 직렬화된 딕셔너리를 만듭니다

        반환값:
            contents (dict): 직렬화된 딕셔너리
        """
        return {
            "review_vocab": self.review_vocab.to_serializable(),
            "rating_vocab": self.rating_vocab.to_serializable(),
        }


# ### `DataLoader`

# In[5]:


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """
    파이토치 DataLoader를 감싸고 있는 제너레이터 함수.
    걱 텐서를 지정된 장치로 이동합니다.
    """
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


# ## `ReviewClassifier` 모델

# In[6]:


class ReviewClassifier(nn.Module):
    """간단한 퍼셉트론 기반 분류기"""

    def __init__(self, num_features):
        """
        매개변수:
            num_features (int): 입력 특성 벡트의 크기
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """분류기의 정방향 계산

        매개변수:
            x_in (torch.Tensor): 입력 데이터 텐서
                x_in.shape는 (batch, num_features)입니다.
            apply_sigmoid (bool): 시그모이드 활성화 함수를 위한 플래그
                크로스-엔트로피 손실을 사용하려면 False로 지정합니다
        반환값:
            결과 텐서. tensor.shape은 (batch,)입니다.
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


# ## 훈련 과정

# ### 설정

# #### 유틸리티

# In[7]:


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# In[8]:


args = Namespace(
    # 날짜와 경로 정보
    frequency_cutoff=25,
    model_state_file="model.pth",
    review_csv="data/yelp/reviews_with_splits_lite.csv",
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir="model_storage/ch3/yelp/",
    vectorizer_file="vectorizer.json",
    # 모델 하이퍼파라미터 없음
    # 훈련 하이퍼파라미터
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # 실행 옵션
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

    print("파일 경로: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

# CUDA 체크
if not torch.cuda.is_available():
    args.cuda = False

print("CUDA 사용여부: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")

# 재현성을 위해 시드 설정
set_seed_everywhere(args.seed, args.cuda)

# 디렉토리 처리
handle_dirs(args.save_dir)


# ### 헬퍼 함수

# In[9]:


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
    """훈련 상태를 업데이트합니다.

    Components:
     - 조기 종료: 과대 적합 방지
     - 모델 체크포인트: 더 나은 모델을 저장합니다

    :param args: 메인 매개변수
    :param model: 훈련할 모델
    :param train_state: 훈련 상태를 담은 딕셔너리
    :returns:
        새로운 훈련 상태
    """

    # 적어도 한 번 모델을 저장합니다
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state["epoch_index"] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]

        # 손실이 나빠지면
        if loss_t >= train_state["early_stopping_best_val"]:
            # 조기 종료 단계 업데이트
            train_state["early_stopping_step"] += 1
        # 손실이 감소하면
        else:
            # 최상의 모델 저장
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])

            # 조기 종료 단계 재설정
            train_state["early_stopping_step"] = 0

        # 조기 종료 여부 확인
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= args.early_stopping_criteria
        )

    return train_state


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


# ### 초기화

# In[10]:


# 만약 코랩에서 실행하는 경우 아래 코드를 실행하여 전처리된 라이트 버전의 데이터를 다운로드하세요.
get_ipython().system("mkdir data")
get_ipython().system("wget https://git.io/JtRSq -O data/download.py")
get_ipython().system("wget https://git.io/JtRSO -O data/get-all-data.sh")
get_ipython().system("chmod 755 data/get-all-data.sh")
get_ipython().run_line_magic("cd", "data")
get_ipython().system("./get-all-data.sh")
get_ipython().run_line_magic("cd", "..")


# In[11]:


if args.reload_from_files:
    # 체크포인트에서 훈련을 다시 시작
    print("데이터셋과 Vectorizer를 로드합니다")
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv, args.vectorizer_file)
else:
    print("데이터셋을 로드하고 Vectorizer를 만듭니다")
    # 데이터셋과 Vectorizer 만들기
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", factor=0.5, patience=1
)
train_state = make_train_state(args)


# ### 훈련 반복

# In[12]:


epoch_bar = tqdm.notebook.tqdm(desc="training routine", total=args.num_epochs, position=0)

dataset.set_split("train")
train_bar = tqdm.notebook.tqdm(
    desc="split=train", total=dataset.get_num_batches(args.batch_size), position=1, leave=True
)
dataset.set_split("val")
val_bar = tqdm.notebook.tqdm(
    desc="split=val", total=dataset.get_num_batches(args.batch_size), position=1, leave=True
)

try:
    for epoch_index in range(args.num_epochs):
        train_state["epoch_index"] = epoch_index

        # 훈련 세트에 대한 순회

        # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
        dataset.set_split("train")
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # 훈련 과정은 5단계로 이루어집니다

            # --------------------------------------
            # 단계 1. 그레이디언트를 0으로 초기화합니다
            optimizer.zero_grad()

            # 단계 2. 출력을 계산합니다
            y_pred = classifier(x_in=batch_dict["x_data"].float())

            # 단계 3. 손실을 계산합니다
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
            loss.backward()

            # 단계 5. 옵티마이저로 가중치를 업데이트합니다
            optimizer.step()
            # -----------------------------------------

            # 정확도를 계산합니다
            acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # 진행 바 업데이트
            train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
            train_bar.update()

        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        # 검증 세트에 대한 순회

        # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
        dataset.set_split("val")
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            # 단계 1. 출력을 계산합니다
            y_pred = classifier(x_in=batch_dict["x_data"].float())

            # 단계 2. 손실을 계산합니다
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # 단계 3. 정확도를 계산합니다
            acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
            val_bar.update()

        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        train_state = update_train_state(args=args, model=classifier, train_state=train_state)

        scheduler.step(train_state["val_loss"][-1])

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()

        if train_state["stop_early"]:
            break

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()
except KeyboardInterrupt:
    print("Exiting loop")


# In[13]:


# 가장 좋은 모델을 사용해 테스트 세트의 손실과 정확도를 계산합니다
classifier.load_state_dict(torch.load(train_state["model_filename"]))
classifier = classifier.to(args.device)

dataset.set_split("test")
batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
running_loss = 0.0
running_acc = 0.0
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # 출력을 계산합니다
    y_pred = classifier(x_in=batch_dict["x_data"].float())

    # 손실을 계산합니다
    loss = loss_func(y_pred, batch_dict["y_target"].float())
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # 정확도를 계산합니다
    acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state["test_loss"] = running_loss
train_state["test_acc"] = running_acc


# In[14]:


print("테스트 손실: {:.3f}".format(train_state["test_loss"]))
print("테스트 정확도: {:.2f}".format(train_state["test_acc"]))


# ### 추론

# In[15]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


# In[16]:


def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """리뷰 점수 예측하기

    매개변수:
        review (str): 리뷰 텍스트
        classifier (ReviewClassifier): 훈련된 모델
        vectorizer (ReviewVectorizer): Vectorizer 객체
        decision_threshold (float): 클래스를 나눌 결정 경계
    """
    review = preprocess_text(review)

    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))

    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)


# In[17]:


test_review = "this is a pretty awesome book"

classifier = classifier.cpu()
prediction = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.5)
print("{} -> {}".format(test_review, prediction))


# ### 해석

# In[18]:


classifier.fc1.weight.shape


# In[19]:


# 가중치 정렬
fc1_weights = classifier.fc1.weight.detach()[0]
_, indices = torch.sort(fc1_weights, dim=0, descending=True)
indices = indices.numpy().tolist()

# 긍정적인 상위 20개 단어
print("긍정 리뷰에 영향을 미치는 단어:")
print("--------------------------------------")
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))

print("====\n\n\n")

# 부정적인 상위 20개 단어
print("부정 리뷰에 영향을 미치는 단어:")
print("--------------------------------------")
indices.reverse()
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))
