import numpy as np
import spacy
import os

from transformers import AutoTokenizer


datasets = np.load(
    os.path.join("../data/cnn_daily/cnn_dm/", f"datasets_test_small.npy"), allow_pickle=True
)
input_sentences = datasets[0]["source"]
print(len(input_sentences))

nlp = spacy.load("en_core_web_sm")

source_txt = [
    " ".join([token.text for token in nlp(sentence) if str(token) != "."]) + "."
    for sentence in input_sentences
]


def get_input_ids(
    tokenizer,
    src_txt,
    bert_compatible_cls=True,
    sep_token=None,
    cls_token=None,
    max_length=None,
):
    sep_token = str(sep_token)
    cls_token = str(cls_token)
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


tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small", use_fast=True)

input_ids = get_input_ids(
    tokenizer,
    source_txt,
    sep_token=tokenizer.sep_token,
    cls_token=tokenizer.cls_token,
    bert_compatible_cls=True,
)
print(len(input_ids))
print(tokenizer.convert_ids_to_tokens(input_ids))
