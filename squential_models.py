import re
import json
import nltk
import keras
import random
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from keras.models import Sequential
from keras.layers import Embedding, Concatenate, Dense, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D, \
    AveragePooling1D, \
    GlobalMaxPool1D, GlobalAveragePooling1D, Flatten, BatchNormalization, Activation, MaxPool1D, concatenate, Input, \
    LSTM
from keras.optimizers import Adam, SGD, rmsprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1
from keras.models import Model
from data_loader import dataLoader


class TextCNNBuilder():

    def __init__(self, vec_dict=None):
        self.tokenizer = Tokenizer(1000)
        self.vec_dict = vec_dict

    def get_long_vec(self, X, X_extra):
        X_long = X.tolist()
        for i in range(len(X_long)):
            X_long.append(list(X_extra)[i])
            return np.array(X_long)

    def process_CNN_word_vector(self):
        data_loader = dataLoader()
        # Without removing stopwords seems to give better performance for CNN
        vec_max_features = 6000
        vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 4), max_df=0.7, max_features=vec_max_features)
        train_docs = data_loader.get_train_docs()
        dev_docs = data_loader.get_dev_docs()
        test_docs = data_loader.get_test_docs()
        X_train = vectorizer.fit_transform(train_docs).toarray()
        X_dev = vectorizer.transform(dev_docs).toarray()
        X_test = vectorizer.transform(test_docs).toarray()
        # print('X_train shape: ', X_train.shape)
        # X_train_vec_mean = data_loader.get_X_train_glove_vec_mean()
        # X_dev_vec_mean = data_loader.get_X_dev_glove_vec_mean()
        # X_test_vec_mean = data_loader.get_X_test_glove_vec_mean()
        # print('X_train_vec_mean shape: ', X_train_vec_mean.shape)
        # X_train = np.concatenate((X_train, X_train_vec_mean), axis=1)
        # X_dev = np.concatenate((X_dev, X_dev_vec_mean), axis=1)
        # X_test = np.concatenate((X_test, X_test_vec_mean), axis=1)

        y_train = data_loader.get_y_train()
        zip_train_X = list(zip(X_train, y_train))
        random.shuffle(zip_train_X)
        X_train, y_train = zip(*zip_train_X)
        X_train, y_train = np.array(list(X_train)), list(y_train)
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        y_dev = data_loader.get_y_dev()
        y_dev_categorical = keras.utils.to_categorical(y_dev, num_classes=2)
        print('X_train shape:', X_train.shape)

        # _activations = ['tanh', 'relu', 'selu']
        # _optimizers = ['sgd', 'adam']
        # _batch_size = [16, 32, 64]
        # params = dict(var_activation=_activations,
        #               var_optimizer=_optimizers,
        #               batch_size=_batch_size)

        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(X_train)
        # maxlen = 1000
        # sequences_train = tokenizer.texts_to_sequences(X_train)
        # sequences_train = pad_sequences(sequences_train, maxlen=maxlen)
        #
        # vocab_size = len(tokenizer.word_index) + 1
        # embedding_size = vec_max_features
        #
        # input_tfidf = Input(shape=(vec_max_features,))
        # input_text = Input(shape=(maxlen,))
        #
        # embedding = Embedding(vocab_size, embedding_size, input_length=maxlen)(input_text)
        # mean_embedding = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1))(embedding)
        # concatenated = concatenate([input_tfidf, mean_embedding])
        #
        # dense1 = Dense(256, activation='relu')(concatenated)
        # dense2 = Dense(32, activation='relu')(dense1)
        # dense3 = Dense(8, activation='sigmoid')(dense2)
        #
        # model = Model(inputs=[input_tfidf, input_text], outputs=dense3)
        # model.summary()
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        num_classes = 2
        verbose, epochs, batch_size = 0, 30, 16
        print('Building model...')
        model = Sequential()
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu', activity_regularizer=l1(0.001)))
        model.add(Dense(64, activation='relu', activity_regularizer=l1(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Fitting model... ")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev_categorical), shuffle=True)
        y_dev_pred = model.predict_classes(X_dev)
        y_test_pred = model.predict_classes(X_test)
        print('Precision on development set: ', precision_score(y_dev, y_dev_pred))
        print('F1-Score on development set: ', f1_score(y_dev, y_dev_pred))
        print('Recall on development set: ', recall_score(y_dev, y_dev_pred))
        print('Negative prediction proportion on development set: ', sum([1 for y in y_dev_pred if y == 0]) / len(y_dev_pred))
        print('Negative prediction proportion on test set: ', sum([1 for y in y_test_pred if y == 0]) / len(y_test_pred))
        return y_test_pred

    def create_cnn_embedding_matrix(self, word_index, embeddings_index, max_length):
        max_features = max_length
        emb_mean, emb_std = -0.005838499, 0.48782197
        all_embs = np.stack(embeddings_index.values())
        embed_size = all_embs.shape[1]
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        count_found = nb_words
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                if word.islower():
                    # try to get the embedding of word in titlecase if lowercase is not present
                    embedding_vector = embeddings_index.get(word.capitalize())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                    else:
                        count_found -= 1
                else:
                    count_found -= 1
        print("Got embedding for ", count_found, " words.")
        return embedding_matrix

    def process_cnn_embedding(self):
        data_loader = dataLoader()
        train_docs = data_loader.get_train_docs()
        y_train = data_loader.get_y_train()

        zip_train_X = list(zip(train_docs, y_train))
        random.shuffle(zip_train_X)
        train_docs, y_train = zip(*zip_train_X)
        train_docs, y_train = list(train_docs), list(y_train)

        dev_docs = data_loader.get_dev_docs()
        test_docs = data_loader.get_test_docs()
        self.tokenizer.fit_on_texts(train_docs+dev_docs+test_docs)
        encoded_docs = self.tokenizer.texts_to_sequences(train_docs)
        # pad documents to a max length of 10000 words
        max_length = 10000
        X_train = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        y_train = keras.utils.to_categorical(y_train, num_classes=2)

        # vocab_size = len(self.tokenizer.word_index) + 1
        glove_vec_dim = 300
        word_index = self.tokenizer.word_index
        embedding_matrix = self.create_cnn_embedding_matrix(word_index, self.vec_dict, max_length)

        emb_mean, emb_std = -0.005838499, 0.48782197
        all_embs = np.stack(self.vec_dict.values())
        embed_size = all_embs.shape[1]
        nb_words = min(max_length, len(word_index))
        embedding_matrix2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        print('Embedding matrix shape: ', embedding_matrix.shape)

        print('Building model...')
        # input1 = Input(shape=(embedding_matrix.shape[0],))
        # embedding1 = Embedding(embedding_matrix.shape[0], glove_vec_dim, weights=[embedding_matrix],
        #               input_length=embedding_matrix.shape[0])(input1)
        # conv1 = Conv1D(filters=16, kernel_size=3, activation='relu', activity_regularizer=l1(0.001))(embedding1)
        # drop1 = Dropout(0.2)(conv1)
        # conv1 = MaxPooling1D(pool_size=2)(drop1)
        #
        # input2 = Input(shape=(embedding_matrix.shape[0],))
        # embedding2 = Embedding(embedding_matrix.shape[0], glove_vec_dim,
        #                        input_length=embedding_matrix.shape[0], trainable=True)(input2)
        # conv2 = Conv1D(filters=16, kernel_size=3, activation='relu',)(embedding2)
        # drop2 = Dropout(0.2)(conv2)
        # conv2 = MaxPooling1D(pool_size=2)(drop2)

        # input2 = Input(shape=(embedding_matrix.shape[0],))
        # embedding2 = Embedding(embedding_matrix.shape[0], glove_vec_dim, trainable=True)(input2)
        # conv2 = Conv1D(filters=32, kernel_size=4, activation='relu', activity_regularizer=l1(0.001))(embedding2)
        # drop1 = Dropout(0.2)(conv2)
        # conv2 = MaxPooling1D(pool_size=2)(drop1)

        # cnn = concatenate([conv1, conv2], axis=-1)
        # flat = Flatten()(cnn)
        # # normal = BatchNormalization()(flat)
        # # x = Dense(128, activation="relu")(flat)
        # x = Dense(2, activation="softmax")(flat)
        # model = Model(inputs=[input1, input2], outputs=x)
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # # plot_model(model, show_shapes=True, to_file='multichannel.png')

        model = Sequential()
        model.add(Embedding(embedding_matrix.shape[0], glove_vec_dim, weights=[embedding_matrix], input_length=embedding_matrix.shape[0]))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=4, kernel_size=3, activation='relu', activity_regularizer=l1(0.01)))
        # model.add(SpatialDropout1D(0.2))
        model.add(MaxPooling1D(pool_size=4))
        # model.add(Conv1D(filters=128, kernel_size=4, activation='relu', activity_regularizer=l1(0.001)))
        # model.add(MaxPooling1D(pool_size=4))
        # model.add(Conv1D(filters=128, kernel_size=4, activation='relu', activity_regularizer=l1(0.001)))
        # model.add(MaxPooling1D(pool_size=4))
        # model.add(LSTM(70))
        # model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        encoded_dev_docs = self.tokenizer.texts_to_sequences(dev_docs)
        X_dev = pad_sequences(encoded_dev_docs, maxlen=max_length, padding='post')
        y_dev = data_loader.get_y_dev()
        y_dev_categorical = keras.utils.to_categorical(y_dev, num_classes=2)
        print("Fitting model... ")
        model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_dev, y_dev_categorical), shuffle=True)
        # evaluate the model
        y_dev_pred = model.predict(X_dev)
        y_dev_pred = np.argmax(y_dev_pred, axis=1)
        # loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
        # print('Accuracy on training set: ', accuracy)
        print('Accuracy on development set: ', accuracy_score(y_dev, y_dev_pred))
        print('Precision on development set: ', precision_score(y_dev, y_dev_pred))
        print('F1-Score on development set: ', f1_score(y_dev, y_dev_pred))
        print('Recall on development set: ', recall_score(y_dev, y_dev_pred))
        print('Zero percentage on development set: ', sum([1 for y in y_dev_pred if y == 0]) / len(y_dev_pred))
        encoded_test_docs = self.tokenizer.texts_to_sequences(test_docs)
        X_test = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')
        y_test_pred = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        # y_test_pred = model.predict_classes(X_test)
        print('Zero percentage on test set: ', sum([1 for y in y_test_pred if y == 0]) / len(y_test_pred))
        return y_test_pred