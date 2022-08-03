import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"{device} is available in torch")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(device)

# encode context the generation is conditioned on
input_ids = tokenizer.encode("I enjoy walking with my cute dog", return_tensors="pt")

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)
print("Output:greedy" + 100 * "-")
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# activate beam search and early_stopping
beam_output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
print("Output:beam-5" + 100 * "-")
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
)
print("Output:beam, n_gram" + 100 * "-")
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True,
)
# now we have 3 output sequences
print("Output:beam, num_seq" + 100 * "-")
for i, beam_output in enumerate(beam_outputs):
    print(f"{i}: {tokenizer.decode(beam_output, skip_special_tokens=True)}")


# set seed to reproduce results. Feel free to change the seed though to get different results
torch.random.manual_seed(42)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=0)
print("Output:top_k=0" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=0, temperature=0.7)
print("Output:temperature" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# set top_k to 50
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50)
print("Output:top_k" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# set top_k to 50 and temperature 0.7
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50, temperature=0.7)
print("Output:top_k, temp" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0)
print("Output:top_p" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids, do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3
)
print("Output:top_p, top_k, num_seq" + 100 * "-")
for i, sample_output in enumerate(sample_outputs):
    print(f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
