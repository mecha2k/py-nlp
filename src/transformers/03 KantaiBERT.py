# Notebook edition (link to original of the reference blogpost [link]
# (https://huggingface.co/blog/how-to-train)).

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

import torch
import transformers
from pathlib import Path
from datasets import load_dataset
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")
print("transformers : ", transformers.__version__)

paths = [str(x) for x in Path("../data/transformers/Kantai").glob("**/*.txt")]
paths = "../data/transformers/Kantai/kant.txt"
print(paths)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=paths,
    vocab_size=52000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)


token_dir = "../data/transformers/Kantai/models/"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
tokenizer.save_model(directory=token_dir)

tokenizer = ByteLevelBPETokenizer(vocab=token_dir + "vocab.json", merges=token_dir + "merges.txt")
print(tokenizer.encode("The Critique of Pure Reason."))
print(tokenizer.encode("The Critique of Pure Reason.").tokens)

print(tokenizer.encode("The Tokenizer."))
print(tokenizer.token_to_id("er"))

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)


config = RobertaConfig(
    vocab_size=52000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
config.to_json_file(token_dir + "config.json")
print(config)


tokenizer = RobertaTokenizer.from_pretrained(token_dir, max_length=512)
model = RobertaForMaskedLM(config=config)
print(model)
print("total parameters : ", model.num_parameters())

num_params = 0
params = list(model.parameters())
for param in params:
    try:
        L2 = len(param[0])
    except TypeError:
        L2 = 1
    num_params += len(param) * L2
print("total parameters : ", num_params)


# dataset = load_dataset("text", data_files="../data/transformers/Kantai/kant.txt")

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/transformers/Kantai/kant.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir=token_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(token_dir, overwrite=True)

fill_mask = pipeline("fill-mask", model=token_dir, tokenizer=token_dir)
print(fill_mask("Human thinking involves<mask>."))
