import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import os
import sys
import json
import math
import copy
import glob
import random
import regex as re
from functools import lru_cache
from tqdm import tqdm, trange
import logging


seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder():
    with open("../data/gpt-2/tf-models/117M/encoder.json", "r") as file:
        encoder = json.load(file)
    with open("../data/gpt-2/tf-models/117M/vocab.bpe", "r", encoding="utf-8") as file:
        bpe_data = file.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


def load_dataset(encoder, path, combine, encoding=None):
    paths = []
    if os.path.isfile(path):
        paths.append(path)
    elif os.path.isdir(path):
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        paths = glob.glob(path)

    raw_text = ""
    token_chunks = []
    for path in tqdm(paths):
        if path.endswith(".npz"):
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            with open(path, "r", encoding=encoding) as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(encoder.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ""
            else:
                raw_text += "<|endoftext|>"
    if raw_text:
        tokens = np.stack(encoder.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler:
    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0, len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk : within_chunk + length]


class GPT2Config:
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


encoder = get_encoder()
out_path = "../data/gpt-2/tf-models/out.npz"
if not os.path.exists(out_path):
    chunks = load_dataset(encoder, "../data/gpt-2/tf-models/dset.txt", 50000, encoding="utf-8")
    print("writing dataset...", out_path)
    np.savez_compressed(out_path, *chunks)


import model

config = GPT2Config()
print(config)
hparams = model.default_hparams()
print(hparams)
# with open(os.path.join("tf-models", args.model_name, "hparams.json")) as f:
#     hparams.override_from_dict(json.load(f))

#     args = parser.parse_args()
#     models_dir = "/content/gpt-2/src/tf-models"
#     enc = encoder.get_encoder(args.model_name, models_dir)
#     hparams = model.default_hparams()
#     with open(os.path.join("tf-models", args.model_name, "hparams.json")) as f:
#         hparams.override_from_dict(json.load(f))
#
#     if args.sample_length > hparams.n_ctx:
#         raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
#
#     if args.model_name == "345M":
#         args.memory_saving_gradients = True
#         if args.optimizer == "adam":
#             args.only_train_transformer_layers = True

# import argparse
# import json
# import os
# import numpy as np
# import tensorflow as tf
# import time
# import tqdm
# from tensorflow.core.protobuf import rewriter_config_pb2
#
# import model, sample, encoder
# from load_dataset import load_dataset, Sampler
# from accumulate import AccumulatingOptimizer
# import memory_saving_gradients
#
# CHECKPOINT_DIR = "checkpoint"
# SAMPLE_DIR = "samples"
#
#
# parser = argparse.ArgumentParser(
#     description="Fine-tune GPT-2 on your custom dataset.",
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# )
#
# parser.add_argument(
#     "--dataset",
#     metavar="PATH",
#     type=str,
#     required=True,
#     help="Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).",
# )
# parser.add_argument(
#     "--model_name", metavar="MODEL", type=str, default="117M", help="Pretrained model name"
# )
# parser.add_argument(
#     "--combine",
#     metavar="CHARS",
#     type=int,
#     default=50000,
#     help="Concatenate input files with <|endoftext|> separator into chunks of this minimum size",
# )
# parser.add_argument(
#     "--encoding", type=str, default="utf-8", help="Set the encoding for reading and writing files."
# )
#
# parser.add_argument("--batch_size", metavar="SIZE", type=int, default=1, help="Batch size")
# parser.add_argument(
#     "--learning_rate", metavar="LR", type=float, default=0.00002, help="Learning rate for Adam"
# )
# parser.add_argument(
#     "--accumulate_gradients",
#     metavar="N",
#     type=int,
#     default=1,
#     help="Accumulate gradients across N minibatches.",
# )
# parser.add_argument(
#     "--memory_saving_gradients",
#     default=False,
#     action="store_true",
#     help="Use gradient checkpointing to reduce vram usage.",
# )
# parser.add_argument(
#     "--only_train_transformer_layers",
#     default=False,
#     action="store_true",
#     help="Restrict training to the transformer blocks.",
# )
# parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer. <adam|sgd>.")
# parser.add_argument(
#     "--noise",
#     type=float,
#     default=0.0,
#     help="Add noise to input training data to regularize against typos.",
# )
#
# parser.add_argument("--top_k", type=int, default=40, help="K for top-k sampling.")
# parser.add_argument(
#     "--top_p", type=float, default=0.0, help="P for top-p sampling. Overrides top_k if set > 0."
# )
#
# parser.add_argument(
#     "--restore_from",
#     type=str,
#     default="latest",
#     help='Either "latest", "fresh", or a path to a checkpoint file',
# )
# parser.add_argument(
#     "--run_name",
#     type=str,
#     default="run1",
#     help="Run id. Name of subdirectory in checkpoint/ and samples/",
# )
# parser.add_argument(
#     "--sample_every", metavar="N", type=int, default=100, help="Generate samples every N steps"
# )
# parser.add_argument(
#     "--sample_length", metavar="TOKENS", type=int, default=1023, help="Sample this many tokens"
# )
# parser.add_argument(
#     "--sample_num", metavar="N", type=int, default=1, help="Generate this many samples"
# )
# parser.add_argument(
#     "--save_every", metavar="N", type=int, default=1000, help="Write a checkpoint every N steps"
# )
#
# parser.add_argument(
#     "--val_dataset",
#     metavar="PATH",
#     type=str,
#     default=None,
#     help="Dataset for validation loss, defaults to --dataset.",
# )
# parser.add_argument(
#     "--val_batch_size", metavar="SIZE", type=int, default=2, help="Batch size for validation."
# )
# parser.add_argument(
#     "--val_batch_count", metavar="N", type=int, default=40, help="Number of batches for validation."
# )
# parser.add_argument(
#     "--val_every",
#     metavar="STEPS",
#     type=int,
#     default=0,
#     help="Calculate validation loss every STEPS steps.",
# )
#
#
# def maketree(path):
#     try:
#         os.makedirs(path)
#     except:
#         pass
#
#
# def randomize(context, hparams, p):
#     if p > 0:
#         mask = tf.random.uniform(shape=tf.shape(context)) < p
#         noise = tf.random.uniform(
#             shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32
#         )
#         return tf.where(mask, noise, context)
#     else:
#         return context
#
#
# def main():
#     args = parser.parse_args()
#     models_dir = "/content/gpt-2/src/tf-models"
#     enc = encoder.get_encoder(args.model_name, models_dir)
#     hparams = model.default_hparams()
#     with open(os.path.join("tf-models", args.model_name, "hparams.json")) as f:
#         hparams.override_from_dict(json.load(f))
#
#     if args.sample_length > hparams.n_ctx:
#         raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
#
#     if args.model_name == "345M":
#         args.memory_saving_gradients = True
#         if args.optimizer == "adam":
#             args.only_train_transformer_layers = True
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
#     with tf.Session(config=config) as sess:
#         context = tf.placeholder(tf.int32, [args.batch_size, None])
#         context_in = randomize(context, hparams, args.noise)
#         output = model.model(hparams=hparams, X=context_in)
#         loss = tf.reduce_mean(
#             tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 labels=context[:, 1:], logits=output["logits"][:, :-1]
#             )
#         )
#
#         if args.val_every > 0:
#             val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
#             val_output = model.model(hparams=hparams, X=val_context)
#             val_loss = tf.reduce_mean(
#                 tf.nn.sparse_softmax_cross_entropy_with_logits(
#                     labels=val_context[:, 1:], logits=val_output["logits"][:, :-1]
#                 )
#             )
#             val_loss_summary = tf.summary.scalar("val_loss", val_loss)
#
#         tf_sample = sample.sample_sequence(
#             hparams=hparams,
#             length=args.sample_length,
#             context=context,
#             batch_size=args.batch_size,
#             temperature=1.0,
#             top_k=args.top_k,
#             top_p=args.top_p,
#         )
#
#         all_vars = [v for v in tf.trainable_variables() if "model" in v.name]
#         train_vars = (
#             [v for v in all_vars if "/h" in v.name]
#             if args.only_train_transformer_layers
#             else all_vars
#         )
#
#         if args.optimizer == "adam":
#             opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
#         elif args.optimizer == "sgd":
#             opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
#         else:
#             exit("Bad optimizer:", args.optimizer)
#
#         if args.accumulate_gradients > 1:
#             if args.memory_saving_gradients:
#                 exit("Memory saving gradients are not implemented for gradient accumulation yet.")
#             opt = AccumulatingOptimizer(opt=opt, var_list=train_vars)
#             opt_reset = opt.reset()
#             opt_compute = opt.compute_gradients(loss)
#             opt_apply = opt.apply_gradients()
#             summary_loss = tf.summary.scalar("loss", opt_apply)
#         else:
#             if args.memory_saving_gradients:
#                 opt_grads = memory_saving_gradients.gradients(loss, train_vars)
#             else:
#                 opt_grads = tf.gradients(loss, train_vars)
#             opt_grads = list(zip(opt_grads, train_vars))
#             opt_apply = opt.apply_gradients(opt_grads)
#             summary_loss = tf.summary.scalar("loss", loss)
#
#         summary_lr = tf.summary.scalar("learning_rate", args.learning_rate)
#         summaries = tf.summary.merge([summary_lr, summary_loss])
#
#         summary_log = tf.summary.FileWriter(os.path.join(CHECKPOINT_DIR, args.run_name))
#
#         saver = tf.train.Saver(var_list=all_vars, max_to_keep=5, keep_checkpoint_every_n_hours=2)
#         sess.run(tf.global_variables_initializer())
#
#         if args.restore_from == "latest":
#             ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name))
#             if ckpt is None:
#                 # Get fresh GPT weights if new run.
#                 ckpt = tf.train.latest_checkpoint(os.path.join("tf-models", args.model_name))
#         elif args.restore_from == "fresh":
#             ckpt = tf.train.latest_checkpoint(os.path.join("tf-models", args.model_name))
#         else:
#             ckpt = tf.train.latest_checkpoint(args.restore_from)
#         print("Loading checkpoint", ckpt)
#         saver.restore(sess, ckpt)
#
#         print("Loading dataset...")
#         chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
#         data_sampler = Sampler(chunks)
#         if args.val_every > 0:
#             if args.val_dataset:
#                 val_chunks = load_dataset(
#                     enc, args.val_dataset, args.combine, encoding=args.encoding
#                 )
#             else:
#                 val_chunks = chunks
#         print("dataset has", data_sampler.total_size, "tokens")
#         print("Training...")
#
#         if args.val_every > 0:
#             # Sample from validation set once with fixed seed to make
#             # it deterministic during training as well as across runs.
#             val_data_sampler = Sampler(val_chunks, seed=1)
#             val_batches = [
#                 [val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
#                 for _ in range(args.val_batch_count)
#             ]
#
#         counter = 1
#         counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, "counter")
#         if os.path.exists(counter_path):
#             # Load the step number if we're resuming a run
#             # Add 1 so we don't immediately try to save again
#             with open(counter_path, "r") as fp:
#                 counter = int(fp.read()) + 1
#
#         def save():
#             maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
#             print("Saving", os.path.join(CHECKPOINT_DIR, args.run_name, "model-{}").format(counter))
#             saver.save(
#                 sess, os.path.join(CHECKPOINT_DIR, args.run_name, "model"), global_step=counter
#             )
#             with open(counter_path, "w") as fp:
#                 fp.write(str(counter) + "\n")
#
#         def generate_samples():
#             print("Generating samples...")
#             context_tokens = data_sampler.sample(1)
#             all_text = []
#             index = 0
#             while index < args.sample_num:
#                 out = sess.run(tf_sample, feed_dict={context: args.batch_size * [context_tokens]})
#                 for i in range(min(args.sample_num - index, args.batch_size)):
#                     text = enc.decode(out[i])
#                     text = "======== SAMPLE {} ========\n{}\n".format(index + 1, text)
#                     all_text.append(text)
#                     index += 1
#             print(text)
#             maketree(os.path.join(SAMPLE_DIR, args.run_name))
#             with open(
#                 os.path.join(SAMPLE_DIR, args.run_name, "samples-{}").format(counter),
#                 "w",
#                 encoding=args.encoding,
#             ) as fp:
#                 fp.write("\n".join(all_text))
#
#         def validation():
#             print("Calculating validation loss...")
#             losses = []
#             for batch in tqdm.tqdm(val_batches):
#                 losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
#             v_val_loss = np.mean(losses)
#             v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
#             summary_log.add_summary(v_summary, counter)
#             summary_log.flush()
#             print(
#                 "[{counter} | {time:2.2f}] validation loss = {loss:2.2f}".format(
#                     counter=counter, time=time.time() - start_time, loss=v_val_loss
#                 )
#             )
#
#         def sample_batch():
#             return [data_sampler.sample(1024) for _ in range(args.batch_size)]
#
#         avg_loss = (0.0, 0.0)
#         start_time = time.time()
#
#         try:
#             while True:
#                 if counter % args.save_every == 0:
#                     save()
#                 if counter % args.sample_every == 0:
#                     generate_samples()
#                 if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
#                     validation()
#
#                 if args.accumulate_gradients > 1:
#                     sess.run(opt_reset)
#                     for _ in range(args.accumulate_gradients):
#                         sess.run(opt_compute, feed_dict={context: sample_batch()})
#                     (v_loss, v_summary) = sess.run((opt_apply, summaries))
#                 else:
#                     (_, v_loss, v_summary) = sess.run(
#                         (opt_apply, loss, summaries), feed_dict={context: sample_batch()}
#                     )
#
#                 summary_log.add_summary(v_summary, counter)
#
#                 avg_loss = (avg_loss[0] * 0.99 + v_loss, avg_loss[1] * 0.99 + 1.0)
#
#                 print(
#                     "[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}".format(
#                         counter=counter,
#                         time=time.time() - start_time,
#                         loss=v_loss,
#                         avg=avg_loss[0] / avg_loss[1],
#                     )
#                 )
#
#                 counter += 1
#         except KeyboardInterrupt:
#             print("interrupted")
#             save()
#
#
# if __name__ == "__main__":
#     main()
