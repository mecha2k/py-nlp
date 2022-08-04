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


def generate_response(question, model, tokenizer, max_len=32):
    user = usr_token + question + sent_token + ""
    encoded = tokenizer.encode(user)
    input_ids = torch.LongTensor(encoded).unsqueeze(dim=0).to(device)
    output = model.generate(
        input_ids,
        max_length=max_len,
        num_beams=10,
        do_sample=False,
        top_k=40,
        no_repeat_ngram_size=2,
        temperature=0.85,
    )
    answer = tokenizer.decode(output[0])
    idx = torch.where(output[0] == tokenizer.encode("<sys>")[0])
    chatbot = tokenizer.decode(output[0][int(idx[0]) + 1 :], skip_special_tokens=True)
    return answer, chatbot


def chatbot(model, user_inputs=None):
    model.to(device)
    model.eval()
    with torch.no_grad():
        user_input_hist = []
        if user_inputs is None:
            while True:
                user_input = input("나 > ").strip()
                user_input_hist.append(user_input)
                if user_input == "quit":
                    break
                print("나 > ", user_input)
                answer, chatbot = generate_response(user_input, model, tokenizer)
                if "답변" in answer:  # 응, 아니 등이 input으로 들어왔을 때
                    answer, chatbot = generate_response(
                        "".join(user_input_hist[-2:]), model, tokenizer
                    )
                print(f"챗봇 > {chatbot.strip()}")
        else:
            for user_input in user_inputs:
                print("나 > ", user_input)
                user_input_hist.append(user_input)
                answer, chatbot = generate_response(user_input, model, tokenizer)
                if "답변" in answer:  # 응, 아니 등이 input으로 들어왔을 때
                    answer, chatbot = generate_response(
                        "".join(user_input_hist[-2:]), model, tokenizer
                    )
                print(f"챗봇 > {chatbot.strip()}")


if __name__ == "__main__":
    model_file = "../data/gpt-2/py-models/kogpt2_chatbot_model.pt"
    if os.path.exists(model_file):
        model = torch.load(model_file)
        print("model loaded from", model_file)
    else:
        model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        print("model not found, original model is used")

    user_inputs = [
        "고민이 많다",
        # 챗봇 > 마음이 예민해져서 매우 괴로우셨겠어요. 저랑 차분하게 잠시 대화를 해봐요.
        "공부하는게 힘들어",
        # 챗봇 > 공부하면 더 많은 선택을 할 수 있죠.
        "그치만 너무 힘든걸",
        # 챗봇 > 조금만 더 버텨보세요.
        "싫어 너무 무기력해",
        # 챗봇 > 평소 즐기던 일이 더 이상 즐겁지 않거나 흥미가 사라졌나요?
        "응 아무것도 하기 싫어",
        # 챗봇 > 아무것도 하고 싶지 않고, 하루가 길게 느껴지나요.
        "맞아",
        # 챗봇 > 00님이 힘든 것 같아 걱정이에요. 더 얘기해볼까요?
        "성적이 자꾸 떨어져",
        # 챗봇 > 노력한 만큼 결과가 나오지 않으면 속상하죠.
        "친구들이랑도 잘 안맞는것같아",
        # 챗봇 > 사람들이 나에게 차갑게 대하는 것 같으신가요?
        "응응",
        # 챗봇 > 자신을 더 사랑해주세요.
        "친구들이 나를 안좋아하는것같아",
        # 챗봇 > 자존감이 떨어져 속상하고 고민이 많이 드셨겠어요.
    ]

    chatbot(model, user_inputs=user_inputs)
