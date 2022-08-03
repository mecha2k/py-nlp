import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"{device} is available in torch")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

context = "Harry Potter is a series of"
label = "seven fantasy novels"
context_input = tokenizer(context)
label_input = tokenizer(label)
label_input["attention_mask"] = [0] * len(label_input["input_ids"])
print(context_input, label_input)

model_input = {
    "input_ids": context_input["input_ids"] + label_input["input_ids"],
    "attention_mask": context_input["attention_mask"] + label_input["attention_mask"],
}
model_input["labels"] = model_input["input_ids"][:]
for i, (l, a) in enumerate(zip(model_input["labels"], model_input["attention_mask"])):
    if a == 1:
        model_input["labels"][i] = -100
print(model_input)

for key in model_input.keys():
    model_input[key] = torch.LongTensor(model_input[key])
print(model_input)

outputs = model(**model_input, return_dict=True)
print(outputs.keys())

model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-4)

for key in model_input.keys():
    model_input[key] = model_input[key].to(device)

for epoch in range(20):
    optim.zero_grad()
    outputs = model(**model_input, return_dict=True)
    loss = outputs["loss"]
    print(loss)
    loss.backward()
    optim.step()


context = "Harry Potter is a series of"
input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)

model.eval()
sample_outputs = model.generate(
    input_ids, do_sample=True, max_length=10, top_k=10, top_p=0.75, num_return_sequences=3
)

print("Output:\n" + 100 * "-")
for i, sample_output in enumerate(sample_outputs):
    print(f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
