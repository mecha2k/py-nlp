import torch
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


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


def chatbot(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        question_hist = []
        while True:
            # question = input("나 > ").strip()
            # question_hist.append(question)
            # if question == "quit":
            #     break
            question = "고민이 많다"
            print("나 > ", question)

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
            print(answer)
            idx = torch.where(output[0] == tokenizer.encode("<sys>")[0])
            # idx = torch.where(output[0] == tokenizer.encode("</d>")[0])
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
                print(answer_new)
                idx = torch.where(output[0] == tokenizer.encode("<sys>")[0])
                chatbot = tokenizer.decode(output[0][int(idx[0]) + 1 :], skip_special_tokens=True)

            print(f"챗봇 > {chatbot.strip()}")


if __name__ == "__main__":
    model_file = "../data/gpt-2/py-models/kogpt2_chatbot_model.pt"
    if os.path.exists(model_file):
        model = torch.load(model_file)
        print("model loaded from", model_file)
    else:
        model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        print("model not found, original model is used")

    chatbot(model)
