# Original file is located at https://colab.research.google.com/drive/1O1gJsEpM7iijBLsy5gsw6um488eL7FG-
# [Hugging Face Models](https://huggingface.co/transformers/model_doc/t5.html)
# [Hugging Face Framework Usage](https://huggingface.co/transformers/usage.html)


import torch
import transformers
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

print(torch.__version__)
print(transformers.__version__)


model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
device = torch.device("cpu")

display_architecture = False
if display_architecture:
    print(model.config)
    print(model)
    print(model.encoder)
    print(model.decoder)
    print(model.forward)


def summarize(text, ml):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    print("Preprocessed and prepared text: \n", t5_prepared_Text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=ml,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


with open("../data/transformers/sample_text.txt", mode="r", encoding="utf-8") as file:
    lines = file.readlines()
print("samples : ", len(lines))

for line in lines:
    print("Number of characters:", len(line), end="\n")
    print("Summarized text: \n", summarize(line, ml=50))
