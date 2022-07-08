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
from pathlib import Path
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

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
tokenizer.save_model(directory=token_dir, prefix="KantaiBERT")

tokenizer = ByteLevelBPETokenizer(
    vocab=token_dir + "KantaiBERT-vocab.json", merges=token_dir + "KantaiBERT-merges.txt"
)
print(tokenizer.encode("The Critique of Pure Reason."))
print(tokenizer.encode("The Critique of Pure Reason.").tokens)


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
print(config)


tokenizer = RobertaTokenizer.from_pretrained("KantaiBERT", max_length=512)
model = RobertaForMaskedLM(config=config)
print(model)
print(model.num_parameters())

LP = list(model.parameters())
lp = len(LP)
print(lp)
for p in range(0, lp):
    print(LP[p])

np = 0
for p in range(0, lp):  # number of tensors
    PL2 = True
    try:
        L2 = len(LP[p][0])  # check if 2D
    except:
        L2 = 1  # not 2D but 1D
        PL2 = False
    L1 = len(LP[p])
    L3 = L1 * L2
    np += L3  # number of parameters per tensor
    if PL2:
        print(p, L1, L2, L3)  # displaying the sizes of the parameters
    if not PL2:
        print(p, L1, L3)  # displaying the sizes of the parameters
print(np)  # total number of parameters


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kant.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./KantaiBERT",
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
trainer.save_model("./KantaiBERT")

fill_mask = pipeline("fill-mask", model="./KantaiBERT", tokenizer="./KantaiBERT")
print(fill_mask("Human thinking involves<mask>."))
