from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
print(loss)
print(prediction_scores)

input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")
outputs = model.generate(input_ids)
outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(outputs)
