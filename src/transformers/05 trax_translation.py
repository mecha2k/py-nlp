# Original file is located at https://colab.research.google.com/drive/15ywNEueTqWj44fajnNELUiDhzZHtg_w0
# [Reference Code](https://colab.research.google.com/github/google/trax/blob/master/trax/intro.ipynb)

import os
import numpy as np
import trax

# Pre-trained model config in gs://trax-ml/models/translation/ende_wmt32k.gin
model = trax.models.Transformer(
    input_vocab_size=33300,
    d_model=512,
    d_ff=2048,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    max_len=2048,
    mode="predict",
)

# @title Initializing the model using pre-trained weights
model.init_from_file("gs://trax-ml/models/translation/ende_wmt32k.pkl.gz", weights_only=True)

# @title Tokenizing a sentence
sentence = "I am only a machine but I have machine intelligence."

tokenized = list(
    trax.data.tokenize(
        iter([sentence]),  # Operates on streams.
        vocab_dir="gs://trax-ml/vocabs/",
        vocab_file="ende_32k.subword",
    )
)[0]

# @title Decoding from the Transformer
tokenized = tokenized[None, :]  # Add batch dimension.
tokenized_translation = trax.supervised.decoding.autoregressive_sample(
    model, tokenized, temperature=0.0
)  # Higher temperature: more diverse results.

# @title De-tokenizing and Displaying the Translation
tokenized_translation = tokenized_translation[0][:-1]  # Remove batch and EOS.
translation = trax.data.detokenize(
    tokenized_translation, vocab_dir="gs://trax-ml/vocabs/", vocab_file="ende_32k.subword"
)
print("The sentence:", sentence)
print("The translation:", translation)
