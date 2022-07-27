# Original file is located at https://colab.research.google.com/drive/1QQGo5vmBkenmNF7WFvOH_s5y7RX1oB3h
# Reference 1 for word embedding:
# https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Reference 2 for cosine similarity:
# SciKit Learn cosine similarity documentation

import nltk
import gensim
import numpy as np
import warnings
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt")
warnings.filterwarnings(action="ignore")

# with open("../data/transformers/text.txt", "r") as file:
#     sample = file.read()
# sample = sample.replace("\n", " ")
# print(len(sample))
#
# sentences = []
# for sentence in sent_tokenize(sample):
#     sentences.append([word.lower() for word in word_tokenize(sentence)])
# sentences = np.array(sentences)
# np.save("../data/transformers/sentences.npy", sentences)
# print(sentences.shape)

# skip_gram = gensim.models.Word2Vec(sentences, min_count=1, vector_size=512, window=5, sg=1)
# skip_gram.save("../data/transformers/skip_gram.model")
# print(skip_gram)

sentences = np.load("../data/transformers/sentences.npy", allow_pickle=True)
print(len(sentences))
print(sentences[0])

skip_gram = gensim.models.Word2Vec.load("../data/transformers/skip_gram.model")
print(skip_gram)
print(len(skip_gram.wv["etext"]))


def word_similarity(model, word1, word2):
    try:
        word1 = model.wv[word1]
        word2 = model.wv[word2]
    except KeyError as e:
        print(e)
        return 0

    # norm1 = np.linalg.norm(word1)
    # norm2 = np.linalg.norm(word2)
    # cosine = np.dot(word1, word2) / (norm1 * norm2)

    return cosine_similarity(word1.reshape(1, -1), word2.reshape(1, -1))[0][0]


word1, word2 = "freedom", "liberty"
print("Case 0: Words in text and dictionary")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "corporations", "rights"
print("Case 1: Word not in text or dictionary")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "etext", "declaration"
print("Case 2: Noisy Relationship")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "justiciar", "judgement"
print("Case 3: Rare words")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "judge", "judgement"
print("Case 4: Replacing words")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "justiciar", "judge"
print("Case 4: Replacing words")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)

word1, word2 = "pay", "debt"
print("Case 5: Entailment")
print("Similarity", word_similarity(skip_gram, word1, word2), word1, word2)
