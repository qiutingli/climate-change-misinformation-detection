import csv
import os
import re
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import nn
import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import keras
from keras.models import Sequential
from keras.optimizers import Adam, SGD, rmsprop
from tensorflow.python.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence

external_docs = {}
with open('twitter_sentiment_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0
    for row in reader:
        if row[0] == '2':
            doc_key = 'train-{}'.format(counter)
            external_docs[doc_key] = {}
            external_docs[doc_key]['text'] = row[1]
            external_docs[doc_key]['label'] = 0
            counter += 1
        elif row[0] == '-1':
            pass

with open('train-external.json', 'a') as file:
    json.dump(external_docs, file)

glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt')

# ============================================================================

num_words = 10
EMBEDDING_DIM = 30
MAX_SEQUENCE_LENGTH = 5
embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, embeddings_initializer='uniform', input_length=MAX_SEQUENCE_LENGTH)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
x_train = sequence.pad_sequences(x_train, maxlen=3)
x_train.shape

corpus = [
    # Positive Reviews

    'This is an excellent movie',
    'The move was fantastic I like it',

    # Negtive Reviews

    "horrible acting",
    'waste of money'
]
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(corpus)
embedded_sentences = word_tokenizer.texts_to_sequences(corpus)
from keras.preprocessing.sequence import pad_sequences
padded_sentences = pad_sequences(embedded_sentences, 7, padding='post')
word_tokenizer.word_index.items()

embeddings_dictionary = dict()
glove_file = open('glove.840B.300d.txt', encoding="utf8")
for line in glove_file:
    try:
        records = line.split()
        word = records[0]
        vector_dimensions = np.array(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    except:
        pass
glove_file.close()

embedding_matrix = np.zeros((17, 300))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        print(word, 'yes')
        embedding_matrix[index-1] = embedding_vector
        print(index)

from keras.layers import Embedding
model = Sequential()
model.add(Embedding(17, 100, input_length=7))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
