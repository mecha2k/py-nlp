import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AdamW, get_scheduler, get_cosine_schedule_with_warmup
from transformers.utils import logging
from tqdm import tqdm

logging.set_verbosity_error()
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
print(tokenizer.all_special_tokens)
print([tokenizer.decode(i) for i in range(10)])


model_file = "../data/gpt-2/py-models/kogpt2_chatbot_model.pt"
if os.path.exists(model_file):
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    print("model loaded from", model_file)
else:
    print("model not found")
    exit(1)


def chatbot():
    with torch.no_grad():
        question_hist = []
        while 1:
            question = input("나 > ").strip()
            question_hist.append(q)

            if question == "quit":
                break
            answer = ""
            user = usr_token + question + sent_token + answer
            encoded = tokenizer.encode(user)
            input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
            output = model.generate(
                input_ids,
                max_length=50,
                num_beams=10,
                do_sample=False,
                top_k=40,
                no_repeat_ngram_size=2,
                temperature=0.85,
            )
            answer = tokenizer.decode(output[0])
            idx = torch.where(output[0] == tokenizer.encode("<sys>")[0])
            chatbot = tokenizer.decode(output[0][int(idx[0]) + 1 :], skip_special_tokens=True)

            if "답변" in answer:  # 응, 아니 등이 input으로 들어왔을 때
                answer_new = ""
                user = (
                    usr_token + "".join(question_hist[-2:]) + sent_token + answer_new
                )  # 직전 history 가지고 와서 sentiment 고려해주기
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(
                    input_ids,
                    max_length=50,
                    num_beams=10,
                    do_sample=False,
                    top_k=40,
                    no_repeat_ngram_size=2,
                    temperature=0.85,
                )
                answer_new = tokenizer.decode(output[0], skip_special_tokens=True)
                idx = torch.where(output[0] == tokenizer.encode("<sys>")[0])
                chatbot = tokenizer.decode(output[0][int(idx[0]) + 1 :], skip_special_tokens=True)

            print(f"챗봇 > {chatbot.strip()}")


if __name__ == "__main__":
    chatbot()
