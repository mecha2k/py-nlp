import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS

print(torch.__version__)
torch.cuda.is_available = lambda: False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


# 2021년 초에 카카오브레인([https://www.kakaobrain.com/](https://www.kakaobrain.com/))에서 다양한 한글 자연어 처리 작업을 위한
# `pororo`('뽀로로'라고 읽습니다)([https://github.com/kakaobrain/pororo](https://github.com/kakaobrain/pororo)) 파이썬
# 라이브러리를 릴리스했습니다. `pororo` 라이브러리는 BERT, Transformer 등 파이토치로 구현된 최신 NLP 모델을 사용해 30여 가지의 자연어
# 처리 작업을 수행합니다. 여기에서는 이 중에 대표적인 몇 가지 작업에 대해서 알아 보겠습니다. `pororo` 라이브러리가 수행할 수 있는 전체
# 작업 목록은 온라인 문서([https://kakaobrain.github.io/pororo/index.html](https://kakaobrain.github.io/pororo/index.html))를
# 참고하세요.
# `pororo`라이브러리는 `pip` 명령으로 간단히 설치할 수 있습니다. 현재는 파이썬 3.6 버전 이상과 파이토치 1.6 버전(CUDA 10.1)을 지원합니다.

print(Pororo.available_tasks())


## 광학 문자 인식
# 먼저 이미지에서 문자를 읽는 광학 문자 인식(Optical Character Recognition) 작업을 수행해 보겠습니다. 광학 문자 인식 작업을
# 수행하려면 `Pororo` 클래스에 `task='ocr'` 매개변수를 지정하여 객체를 만듭니다.

ocr = Pororo(task="ocr")
print(ocr("../data/ocr-test.png"))


# <핸즈온 머신러닝>에서 세로로 쓰여진 '2판'은 인식을 못했고 <GAN 인 액션>과 <미술관에 GAN 딥러닝>은 행을 조금 혼동하고 있지만 전반적으로
# 높은 인식율을 보여주고 있습니다. 광학 문자 인식 작업에 지원하는 언어는 영어와 한국어입니다. 지원하는 언어 목록을 보려면 `pororo` 패키지의
# 온라인 문서를 참고하세요. 현재는 `Pororo` 클래스에서 가능한 언어를 직접 확인할 수는 없습니다. 다만 다음처럼 `SUPPORTED_TASKS` 딕셔너리에
# 매핑된 광학 문자 인식 클래스의 `get_available_langs()` 정적 메서드를 호출하여 확인할 수 있습니다.

SUPPORTED_TASKS["ocr"].get_available_langs()


# 로컬에 있는 파일 뿐만 아니라 URL을 전달할 수도 있습니다. 다음과 같이 영어로 쓰여진
# 표지판([https://bit.ly/london-sign](https://bit.ly/london-sign), Goldflakes, CC BY-SA 4.0)을 인식해 보죠.

print(ocr("https://bit.ly/london-sign", detail=True))

# 결과에서 알 수 있듯이 이미지 구역에 따라 인식한 글씨를 나누어 리스트로 반환하고 있습니다. 또한 `detail=True`로 지정하면 인식된 글자
# 구역의 왼쪽 위에서 시계 방향으로 4개의 사각형 모서리 좌표를 반환합니다. `pororo`의 광학 인식 문자에 사용되는 OCR 모델은 내부 데이터와
# AI hub의 한국어 글자체 이미지 AI 데이터([https://www.aihub.or.kr/aidata/133](https://www.aihub.or.kr/aidata/133))을 사용하여
# 훈련되었습니다.

## 이미지 캡셔닝
# 이미지 캡셔닝(image captioning)은 이미지를 설명하는 텍스트를 만드는 작업입니다. `pororo`의 이미지 캡션은 한국어, 영어, 중국어,
# 일본어를 지원합니다. 가능한 언어 목록을 확인해 보죠. 이미지 캡셔닝 작업은 `'caption'`으로 지정합니다.

SUPPORTED_TASKS["caption"].get_available_langs()


# 광학 문자 인식과 마찬가지로 `task` 매개변수에 `'caption'`으로 지정하고 `lang='ko'`으로 지정하여 한글 캡션을 위한 객체를 만들어 보겠습니다.
caption = Pororo(task="caption", lang="ko")

# `Pororo` 클래스는 새로운 객체를 만들 때마다 사용할 모델을 다운로드하여 로드합니다. 다운로드된 데이터는 리눅스일 경우 `~/.pororo` 아래
# 저장되고 윈도우의 경우 `C:\\pororo` 아래 저장하여 나중에 재사용합니다. 다음과 같은 이미지([http://bit.ly/ny-timesquare]
# (http://bit.ly/ny-timesquare), Terabass, CC BY-SA 3.0)의 캡션을 만들어 보겠습니다.
# print(caption("../data/New_york_times_square-terabass.jpg"))

# 이번에는 영어로 캡션을 만들어 보겠습니다.
caption = Pororo(task="caption", lang="en")
print(caption("http://bit.ly/ny-timesquare"))


# 각 작업이 사용하는 모델은 `Pororo` 클래스의 `available_models()` 정적 메서드에서 얻을 수 있습니다. 이 메서드를 호출할 때 `task`
# 매개변수에 작업 이름을 지정합니다.

Pororo.available_models(task="caption")

# 또는 앞에서와 같이 `SUPPORTED_TASKS` 딕셔너리 객체를 사용해 얻은 클래스의 `get_available_models()` 메서드를 호출할 수도 있습니다.

SUPPORTED_TASKS["caption"].get_available_models()


# 사용하는 모델 목록에서 볼 수 있듯이 이미지 캡셔닝은 트랜스포머 기반의 영어 모델만 사용합니다. 그외 언어에 대해서는 사실 다음 절에서
# 설명할 기계 번역 모델을 통해 번역한 것입니다.

## 기계 번역
# 기계 번역 작업은 페이스북에서 만든 fairseq의 TransformerModel
# (https://fairseq.readthedocs.io/en/latest/models.html#module-fairseq.models.transformer) 모델을 사용합니다. 훈련은 내부
# 데이터를 사용했고 테스트 데이터는 Multilingual TED Talk(http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/) 데이터를
# 사용했습니다.
# 번역 작업은 `'translation'`이며 `lang` 매개변수는 `multi`로 지정하며 한국어, 영어, 일본어, 중국어를 번역할 수 있습니다. 또한
# `model` 매개변수에 사용할 모델을 지정할 수 있습니다. 기본적으로 인코더와 디코더가 각각 6개의 층으로 이루어진
# `'transformer.large.multi.mtpg'`을 사용합니다. 기본 모델을 사용해 기계 번역 객체를 만들어 보겠습니다.

mt = Pororo(task="translation", lang="multi")


# 한국어 샘플 텍스트를 영어로 번역해 보겠습니다. 원본 텍스트의 언어는 `src`에 지정하고 번역하려는 타깃 언어는 `tgt`에 지정합니다.
# 지정할 수 있는 옵션은 앞에서 언급한 4개의 언어인 `'ko'`, `'en'`, `'ja'`, `'zh'`입니다.
text1 = (
    "퍼서비어런스(Perseverance)는 화성 탐사차로 2020년 7월 30일 발사하여 2021년 2월 18일 화성에 착륙하였다. 화성의 생명체 거주 "
    "여부, 화성의 고대 환경 조사, 화성 지표의 역사 등을 밝히는 것이 이 탐사선의 목표다. 더불어 중요한 목표는 미래의 인류가 화성을 "
    "유인 탐사할 때 위험한 것이 없는지 탐색하고, 대기의 조성을 알려주어 미래의 기지를 건설하는 데 도움을 주는 것이다. 또 인간이 "
    "어떤 조건으로 착륙해야 되는지 등을 탐색한다. 예산은 원래 15억 달러를 배정했는데, 지금은 더 늘어나서 25억 달러다. "
    "특이사항으로는 인사이트가 마스 큐브 원과 화성에 함께 갔던 것과 비슷하게 인제뉴어티와 함께 발사되었다. 또한 큐리오시티의 디자인을"
    " 많이 재사용했다. 따라서 새로운 기술보다는 이전 로버들의 좋은 점을 합친 것이라고 보면 된다. 참고로, 마스 2020(Mars 2020)은"
    " 퍼서비어런스와 인제뉴어티 드론 헬리콥터를 포함한, NASA의 화성 지표면 로봇 탐사 계획의 명칭이다. 최초로 화성의 바람소리를 녹음했다."
)
print(text1)
print(mt(text1, src="ko", tgt="en"))


# 몇몇 번역 오류가 있지만 대체적으로 번역 결과는 좋습니다.
# 기본 모델 외에 약간의 성능을 희생하면서 2배 정도 빠른 속도를 내는 `'transformer.large.multi.fast.mtpg'` 모델을 지정할 수 있습니다.
# 이 모델은 12개의 인코더 층과 1개의 디코더 층으로 이루어집니다. 이 모델을 사용해 앞에서와 동일한 텍스트를 번역해 보죠.

mt_fast = Pororo(task="translation", lang="multi", model="transformer.large.multi.fast.mtpg")
print(mt_fast(text1, src="ko", tgt="en"))


## 텍스트 요약
# 텍스트 요약(text summarization)은 비교적 긴 텍스트를 짧은 문장 몇개로 압축하여 표현하는 작업입니다. `pororo`는 텍스트 요약을 위해
# 3개의 모델을 제공합니다.
# 먼저 `abstractive` 모델은 하나의 완전한 문장으로 텍스트 내용을 요약합니다. 이 모델은 SKT에서 개발한 KoBART(https://github.com/SKT-AI/KoBART)
# 모델을 사용합니다. 학습에 사용한 데이터는 데이콘(DACON)의 문서 추출요약 경진 대회 데이터
# (https://dacon.io/competitions/official/235671/data/)와 AI 허브에서 공개한 AI 학습용
# 데이터(https://www.aihub.or.kr/node/9176)입니다.
# `task='summary'`로 지정하여 텍스트 요약 작업임을 알리고 `lang='ko'`로 지정합니다. 현재 텍스트 요약 작업은 한글만 지원합니다.
# 먼저 `abstractive` 모델을 사용해 보죠.

# **경고**: 최신 `transformers` 패키지에서 나눗셈 에러가 발생할 수 있습니다. 이런 경우 다음 명령으로 `transformers` 4.7.0 버전을 설치해 주세요.
abs_summ = Pororo(task="summary", lang="ko", model="abstractive")
print(abs_summ(text1))

# 꽤 잘 요약된 것 같습니다. 다음에는 `bullet` 모델을 사용해 보겠습니다. 이 모델은 짧은 몇 개의 문장으로 텍스트를 요약합니다.
bul_summ = Pororo(task="summary", lang="ko", model="bullet")
print(bul_summ(text1))


# 텍스트 요약 작업은 여러 개의 텍스트를 파이썬 리스트로 만들어 전달하면 한 번에 여러 텍스트를 요약할 수 있습니다. 예를 위해 두 번째
# 텍스트 샘플을 만들어 `text1`과 함께 전달해 보겠습니다.

text2 = (
    "알로사우루스(라틴어: Allosaurus)는 후기 쥐라기(1억 5600만 년 전 ~ 1억 4500만 년 전)를 대표하는 큰 육식공룡이다. "
    "알로사우루스라는 학명의 어원은 고대 그리스어 (그리스어: αλλοςσαυρος)인데, 이 말은 '특이한 도마뱀'이라는 뜻으로, 한자표기 "
    "이특룡(異特龍)은 여기서 비롯되었다. 미국의 고생물학자 오스니얼 찰스 마시가 알로사우루스속 화석을 처음으로 기재했다. 수각류 "
    "공룡 중 비교적 초기에 알려진 공룡 중 하나로, 그동안 여러 영화나 다큐멘터리에도 많이 등장했다. 알로사우루스는 짧은 목에 큰 머리, "
    "긴 꼬리와 짧은 앞다리를 가진 전형적으로 거대한 수각류 공룡이다. 생존 당시에는 대형 포식자로서 먹이사슬의 최고점에 있었다. "
    "보통 아성체나 소형 용각류, 조각류, 검룡류와 같은 대형 초식공룡을 사냥했을 것으로 추정된다. 아성체나 소형 용각류 등을 사냥할 때 "
    "무리를 지어 조직적으로 사냥했다는 추정이 있지만, 이들이 사회적이라는 증거는 많지 않으며 심지어 자신들끼리 싸움을 했을 수도 있다. "
    "매복하다가 입을 크게 벌리고 윗턱을 손도끼 내리치듯이 가격해 큰 사냥감을 잡았을 것으로 생각된다."
)

print(bul_summ([text1, text2]))


# 이번에는 텍스트에서 가중 중요한 3개의 문장을 추출하는 `extractive` 모델을 사용해 보겠습니다. 이 모델은 페이스북의
# RoBERTa(https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) 모델을
# 위에서 언급한 말뭉치에서 훈련시켜 사용합니다.

ext_summ = Pororo(task="summary", lang="ko", model="extractive")
print(ext_summ(text1))

# 마찬가지로 여러 개의 문장을 리스트로 묶어서 전달할 수 있으며 `return_list=True`로 지정하면 추출한 3개의 문장을 리스트로 만들어 반환합니다.
print(ext_summ([text1, text2], return_list=True))


## 감성 분석
# 감성 분석(sentiment analysis)는 텍스트를 긍정과 부정으로 분류하는 작업입니다. `pororo`는 페이스북의 RoBERTa 모델을 네이버 쇼핑
# 리뷰 데이터셋([https://github.com/bab2min/corpus/tree/master/sentiment](https://github.com/bab2min/corpus/tree/master/sentiment))과
# 네이버 영화 리뷰 데이터셋([https://github.com/e9t/nsmc](https://github.com/e9t/nsmc))에서 훈련한 모델을 제공합니다.
# 또한 일본어를 위한 모델도 제공합니다.
# 먼저 네이버 쇼핑 리뷰 데이터셋에서 훈련한 모델(`model='brainbert.base.ko.shopping'`)을 사용해 보죠. 이 모델은 네이버 쇼핑 리뷰
# 데이터셋에서 약 95%의 정확도를 달성했습니다.

sa_shop = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")
print(sa_shop("정말 혼자 공부하기 너무 좋은 머신러닝 독학 책"))

# 일반적인 텍스트의 감성은 쉽게 분류합니다. 하지만 비유적인 표현은 쉽게 감지하기 어렵습니다. 아래 텍스트는 달팽이에 비유하여 느린 배송
# 속도를 비꼬는 말이지만 긍정적으로 분류했습니다.
print(sa_shop("달팽이 같은 놀라운 배송 속도"))

# 쇼핑 리뷰 데이터를 사용해 훈련했기 때문에 비교적 쇼핑과 관련된 텍스트에 담긴 감정은 잘 잡아내지만 영화에 대한 것은 그렇지 못합니다.
print(sa_shop("택배 속도 놀랍군"))
print(sa_shop("반전을 거듭하는데 와.."))


# 이번에는 네이터 영화 리뷰 데이터셋에서 훈련한 모델을 사용해 보겠습니다. `model='brainbert.base.ko.nsmc'`로 지정합니다. 이 모델은
# 네이버 영화 리뷰 데이터셋에서 약 90%의 정확도를 냈습니다.
sa_movie = Pororo(task="sentiment", model="brainbert.base.ko.nsmc", lang="ko")

# 앞에서와 같이 동일한 예를 적용해 보죠. 여기에서는 반대로 영화와 관련된 감정은 잘 감지하지만 택배에 대해서는 그렇지 못합니다.
print(sa_movie("택배 속도 놀랍군"))
print(sa_movie("반전을 거듭하는데 와.."))


## 자연어 추론
# 자연어 추론(natural language inference, NLI)는 두 문장 사이의 관계가 함의(entailment), 모순(contradiction), 중립(neutral)인지
# 추론합니다. `pororo`에서 자연어 추론 작업은 `'nli'`로 지정하며 RoBERTa 구조를 사용해 한국어, 영어, 일본어, 중국어 데이터셋에서
# 훈련한 모델을 제공합니다.

SUPPORTED_TASKS["nli"].get_available_langs()


# `lang='ko'`로 지정하여 간단한 한국어 문장을 추론해 보겠습니다.
nli = Pororo(task="nli", lang="ko")

# 아래 3개의 예에서 처음 2개는 함의, 모순 관계를 잘 감지했습니다. 하지만 마지막 3번째 예는 함의가 아니라 중립으로 출력된 것을 알 수
# 있습니다. 이 모델은 두 문장의 인과 관계를 감지하기 쉽지 않아 보입니다.
print(nli("비가 온다", "날씨가 우중충하다"))
print(nli("비가 온다", "구름 사이로 햇살이 비친다"))
print(nli("비가 온다", "옷이 비에 젖다"))


## 제로샷 토픽 분류
# 마지막으로 제로샷 토픽 분류(zero-shot topic classification) 작업을 알아 보겠습니다. 제로샷 토픽 분류는 주어진 텍스트를 훈련에서
# 사용하지 않은 처음 본 클래스 레이블에 할당할 수 있습니다. 제로샷 토픽 분류 작업은 `'zero-topic'`으로 지정하며 자연어 추론과
# 마찬가지로 한국어, 영어, 중국어, 일본어를 지원합니다. 먼저 제로샷 토픽 분류를 위한 객체를 만들어 보겠습니다.

zsl = Pororo(task="zero-topic", lang="ko")


# `zsl` 객체를 호출할 때 분류하려는 대상 문장을 첫 번째 매개변수로 전달하고 분류 토픽 리스트를 두 번째 매개변수로 전달합니다.
print(zsl("손흥민이 골을 넣었다", ["정치", "사회", "스포츠", "연예"]))

# 출력 결과를 보면 '스포츠'에 대한 점수가 높게 나왔으므로 올바르게 분류가 되었습니다. 사실 제로샷 토픽 분류는 이전에 살펴 보았던
# 자연어 처리 모델을 사용합니다.
# 먼저 '손흥민이 골을 넣었다'와 '이 문장은 정치에 관한 것이다'라는 두 문장의 자연어 추론을 수행합니다. 그다음 나머지 '사회', '스포츠',
# '연예' 레이블에 대해서도 같은 작업을 반복하여 4번의 자연어 추론을 수행합니다. 각 수행 결과에서 중립(neutral)을 빼고
# 모순(contradiction)과 함의(entailment) 결과를 소프트맥스 함수를 통과시켜 확률로 변환합니다. `zsl` 객체가 반환한 결과는 이렇게 각
# 레이블에 대해 수행한 후 얻은 함의(entailment)에 대한 확률값입니다.

# 제로샷 토픽 분류가 자연어 이해를 사용하므로 토픽에 국한하지 않고 첫 번째 문장이 어떤 상황에 관한 문장인지를 파악하는데 사용할 수도 있습니다.
print(zsl("손흥민이 골을 넣었다", ["공격", "수비", "승리", "패배", "경기", "실수"]))

# 제로샷 토픽 분류는 한글 외에 다른 언어에 대해서는 결과를 잘못 반환하는 버그가 있습니다. 자세한 내용은 깃허브
# 이슈([https://github.com/kakaobrain/pororo/issues/52](https://github.com/kakaobrain/pororo/issues/52))를 참고하세요.
