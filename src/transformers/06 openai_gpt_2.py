# Original file is located at https://colab.research.google.com/drive/1ENY-Rd_eWkXd9SyDn3lAuWtOokzV8VWY
#
# #OpenAI GTP-2
# Copyright 2020, Denis Rothman MIT License. Denis Rothman created the Colab notebook using the OpenAI
# repository, adding title steps for educational purposes only.
#
# It is important to note that we are running a low-level GPT-2 model
# and not a one-line call to obtain a result. We are also
# avoiding pre-packaged versions. We are getting our hands dirty to
# understand the architecture of a GPT-2 from scratch. You might get
# some deprecation messages. However, the effort is worthwhile.
#
# ***Code Reference***
# [Reference: OpenAI Repository](https://github.com/openai/gpt-2)
#
# ***Model Reference***
# [Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever,2019,'Language Models are Unsupervised Multitask Learners'](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
#
#
# Step 1: Pre-requisite: activate GPU in the notebook settings runTime menu

# @title Step 2: Cloning the OpenAI GPT-2 Repository
# !git clone https://github.com/openai/gpt-2.git

# @title Step 3: Installing the requirements
# import os                     # when the VM restarts import os necessary
# os.chdir("/content/gpt-2")
# !pip3 install -r requirements.txt

# Commented out IPython magic to ensure Python compatibility.
# @title Step 4 Checking the Version of TensorFlow
# Colab has tf 1.x and tf 2.x installed
# Restart runtime using 'Runtime' -> 'Restart runtime...'
# %tensorflow_version 1.x
import tensorflow as tf

print(tf.__version__)

# @title Step 5: Downloading the 345M parameter GPT-2 Model
# run code and send argument
import os  # after runtime is restarted

# os.chdir("/content/gpt-2")
# !python3 download_model.py '345M'

# @title Step 6: Printing UTF encoded text to the console
# !export PYTHONIOENCODING=UTF-8

# @title Step 7: Project Source Code
import os  # import after runtime is restarted

os.chdir("/content/gpt-2/src")

# @title Step 7a: Interactive Conditional Samples (src)
# Project Source Code for Interactive Conditional Samples:
# /content/gpt-2/src/interactive_conditional_samples.py file
import json
import os
import numpy as np
import tensorflow as tf

# @title Step 7b: Importing model sample encoder
import model, sample, encoder

# if following message:
# ModuleNotFoundError: No module named 'tensorflow.contrib'
# then go back and run Step 2 Checking TensorFlow version

# @title Step 8: Defining the model
def interact_model(model_name, seed, nsamples, batch_size, length, temperature, top_k, models_dir):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, "hparams.json")) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print("Prompt should not be empty!")
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(
                    output, feed_dict={context: [context_tokens for _ in range(batch_size)]}
                )[:, len(context_tokens) :]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


# @title Step 9: Interacting with GPT-2
interact_model("345M", None, 1, 1, 300, 1, 0, "/content/gpt-2/models")
