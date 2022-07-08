import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from transformers import logging

logging.set_verbosity_error()

# [Reference Notebook by Chris McCormick and Nick Ryan]
# (https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX)
# [Reference Article by Chris McCormick and Nick Ryan]
# (https://mccormickml.com/2019/07/22/BERT-fine-tuning/)


# device_name = tf.test.gpu_device_name()
# if device_name != "/device:GPU:0":
#     raise SystemError("GPU device not found")
# print("Found GPU at: {}".format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")
# n_gpu = torch.cuda.device_count()
# print(torch.cuda.get_device_name(0))

# source of dataset : https://nyu-mll.github.io/CoLA/
df = pd.read_csv(
    "../data/transformers/in_domain_train.tsv",
    delimiter="\t",
    header=None,
    names=["sentence_source", "label", "label_notes", "sentence"],
)
print(df.info())


sentences = df.sentence.values
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print("Tokenize the first sentence:")
print(tokenized_texts[0])

# Set the maximum sequence length. The longest sequence in our training set is 47,
maxlen = 128

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=2018, test_size=0.1
)
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, input_ids, random_state=2018, test_size=0.1
)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=batch_size
)


configuration = BertConfig()

model = BertModel(configuration)
configuration = model.config
print(configuration)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# model.cuda()


# This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

# Don't apply weight decay to any parameters whose names include these tokens.
# (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.weight"]
# Separate the `weight` parameters from the `bias` parameters.
# - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
# - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
optimizer_grouped_parameters = [
    # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
    {
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.1,
    },
    # Filter for parameters which *do* include those.
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0,
    },
]


# # @title The Hyperparemeters for the Training Loop
# # optimizer = BertAdam(optimizer_grouped_parameters,
# #                      lr=2e-5,
# #                      warmup=.1)
#
# # Number of training epochs (authors recommend between 2 and 4)
# epochs = 4
#
# optimizer = AdamW(
#     optimizer_grouped_parameters,
#     lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
#     eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
# )
# # Total number of training steps is number of batches * number of epochs.
# # `train_dataloader` contains batched data so `len(train_dataloader)` gives
# # us the number of batches.
# total_steps = len(train_dataloader) * epochs
#
# # Create the learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=0, num_training_steps=total_steps  # Default value in run_glue.py
# )
#
#
# # In[43]:
#
#
# # Creating the Accuracy Measurement Function
# # Function to calculate the accuracy of our predictions vs labels
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
#
# # In[44]:
#
#
# # @title The Training Loop
# t = []
#
# # Store our loss and accuracy for plotting
# train_loss_set = []
#
# # trange is a tqdm wrapper around the normal python range
# for _ in trange(epochs, desc="Epoch"):
#
#     # Training
#
#     # Set our model to training mode (as opposed to evaluation mode)
#     model.train()
#
#     # Tracking variables
#     tr_loss = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#
#     # Train the data for one epoch
#     for step, batch in enumerate(train_dataloader):
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Clear out the gradients (by default they accumulate)
#         optimizer.zero_grad()
#         # Forward pass
#         outputs = model(
#             b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels
#         )
#         loss = outputs["loss"]
#         train_loss_set.append(loss.item())
#         # Backward pass
#         loss.backward()
#         # Update parameters and take a step using the computed gradient
#         optimizer.step()
#
#         # Update the learning rate.
#         scheduler.step()
#
#         # Update tracking variables
#         tr_loss += loss.item()
#         nb_tr_examples += b_input_ids.size(0)
#         nb_tr_steps += 1
#
#     print("Train loss: {}".format(tr_loss / nb_tr_steps))
#
#     # Validation
#
#     # Put model in evaluation mode to evaluate loss on the validation set
#     model.eval()
#
#     # Tracking variables
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#
#     # Evaluate data for one epoch
#     for batch in validation_dataloader:
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Telling the model not to compute or store gradients, saving memory and speeding up validation
#         with torch.no_grad():
#             # Forward pass, calculate logit predictions
#             logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#         # Move logits and labels to CPU
#         logits = logits["logits"].detach().cpu().numpy()
#         label_ids = b_labels.to("cpu").numpy()
#
#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#
#         eval_accuracy += tmp_eval_accuracy
#         nb_eval_steps += 1
#
#     print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
#
# # In[45]:
#
#
# # @title Training Evaluation
# plt.figure(figsize=(15, 8))
# plt.title("Training loss")
# plt.xlabel("Batch")
# plt.ylabel("Loss")
# plt.plot(train_loss_set)
# plt.show()
#
# # In[46]:
#
#
# # @title Predicting and Evaluating Using the Hold-out Dataset
# df = pd.read_csv(
#     "out_of_domain_dev.tsv",
#     delimiter="\t",
#     header=None,
#     names=["sentence_source", "label", "label_notes", "sentence"],
# )
#
# # Create sentence and label lists
# sentences = df.sentence.values
#
# # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
# sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
# labels = df.label.values
#
# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
#
# MAX_LEN = 128
#
# # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# # Pad our input tokens
# input_ids = pad_sequences(
#     input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
# )
# # Create attention masks
# attention_masks = []
#
# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#     seq_mask = [float(i > 0) for i in seq]
#     attention_masks.append(seq_mask)
#
# prediction_inputs = torch.tensor(input_ids)
# prediction_masks = torch.tensor(attention_masks)
# prediction_labels = torch.tensor(labels)
#
# batch_size = 32
#
# prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
# prediction_sampler = SequentialSampler(prediction_data)
# prediction_dataloader = DataLoader(
#     prediction_data, sampler=prediction_sampler, batch_size=batch_size
# )
#
# # In[47]:
#
#
# # Prediction on test set
#
# # Put model in evaluation mode
# model.eval()
#
# # Tracking variables
# predictions, true_labels = [], []
#
# # Predict
# for batch in prediction_dataloader:
#     # Add batch to GPU
#     batch = tuple(t.to(device) for t in batch)
#     # Unpack the inputs from our dataloader
#     b_input_ids, b_input_mask, b_labels = batch
#     # Telling the model not to compute or store gradients, saving memory and speeding up prediction
#     with torch.no_grad():
#         # Forward pass, calculate logit predictions
#         logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#     # Move logits and labels to CPU
#     logits = logits["logits"].detach().cpu().numpy()
#     label_ids = b_labels.to("cpu").numpy()
#
#     # Store predictions and true labels
#     predictions.append(logits)
#     true_labels.append(label_ids)
#
# # In[48]:
#
#
# # @title Evaluating Using Matthew's Correlation Coefficient
# # Import and evaluate each test batch using Matthew's correlation coefficient
# from sklearn.metrics import matthews_corrcoef
#
# matthews_set = []
#
# for i in range(len(true_labels)):
#     matthews = matthews_corrcoef(true_labels[i], np.argmax(predictions[i], axis=1).flatten())
#     matthews_set.append(matthews)
#
# # The final score will be based on the entire test set, but let's take a look at the scores on the individual batches to get a sense of the variability in the metric between batches.
# #
#
# # In[49]:
#
#
# # @title Score of Individual Batches
# matthews_set
#
# # In[50]:
#
#
# # @title Matthew's Evaluation on the Whole Dataset
# # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
# flat_predictions = [item for sublist in predictions for item in sublist]
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# flat_true_labels = [item for sublist in true_labels for item in sublist]
# matthews_corrcoef(flat_true_labels, flat_predictions)
