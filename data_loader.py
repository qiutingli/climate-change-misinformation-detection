import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from data_preprocess import DataPreprocessor

class dataLoader():

    def __init__(self, vec_dict=None):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 6), max_df=0.6, max_features=12000)
        self.tokenizer = Tokenizer(1000)
        self.data_preprocessor = DataPreprocessor()
        self.vec_dict = vec_dict
        self.train_file_names = ['train',
                                 # 'train-external-skeptical-misinfo', 'train-external-skeptical-truth',
                                 'train-external-news', 'train-external-bbc', 'train-external-bbc-2',
                                 # 'train-external-papers', 'train-external-wiki', 'train-external'
                                 ]


    def view_dataset(self, file_name):
        with open('data/{}.json'.format(file_name)) as file:
            data_json = json.load(file)
            for key in data_json.keys():
                print('Key in json: ', key)
                print('Value:', data_json[key]['label'])
                break
            print('File length:', len(data_json.keys()))

    def get_label_text_from_file(self, file_name):
        labels = []
        texts = []
        with open('data/{}.json'.format(file_name)) as file:
            data_json = json.load(file)
            for key in data_json.keys():
                labels.append(data_json[key]['label'])
                texts.append(data_json[key]['text'])
        return labels, texts

    def get_train_docs(self):
        docs = []
        for file_name in self.train_file_names:
            file_docs = self.get_label_text_from_file(file_name)[1]
            docs.extend(file_docs)
        cleaned_docs = []
        for text in docs:
            text = self.data_preprocessor.clean(text)
            cleaned_docs.append(text)
        return cleaned_docs

    def get_dev_docs(self):
        docs = []
        with open('data/dev.json') as file:
            data_json = json.load(file)
            for key in data_json.keys():
                docs.append(data_json[key]['text'])
        cleaned_docs = []
        for text in docs:
            text = self.data_preprocessor.clean(text)
            cleaned_docs.append(text)
        return cleaned_docs

    def get_test_docs(self):
        docs = []
        with open('data/test-unlabelled.json') as file:
            data_json = json.load(file)
            for key in data_json.keys():
                docs.append(data_json[key]['text'])
        cleaned_docs = []
        for text in docs:
            text = self.data_preprocessor.clean(text)
            cleaned_docs.append(text)
        return cleaned_docs

    def get_X_train_tfidf(self):
        train_docs = self.get_train_docs()
        X_train = self.vectorizer.fit_transform(train_docs)
        return X_train

    def get_X_dev_tfidf(self):
        dev_docs = self.get_dev_docs()
        return self.vectorizer.transform(dev_docs)

    def get_X_test_tfidf(self):
        test_docs = self.get_test_docs()
        return self.vectorizer.transform(test_docs)

    def vectorize_text_glove(self, tokenized_text):
        preprocessed_tokens = tokenized_text
        word_vectors = []
        for token in preprocessed_tokens:
            try:
                word_vectors.append(self.vec_dict[token])
            except:
                word_vectors.append([0]*300)
        return word_vectors

    def get_X_train_glove_vec_mean(self):
        train_docs = self.get_train_docs()
        preprocessor = DataPreprocessor()
        tokenized_texts = preprocessor.preprocess_data(train_docs)
        X_train = np.empty((0, 300), float)
        for tokenized_text in tokenized_texts:
            if len(tokenized_text) == 0:
                print('Empty tokenized_text')
                # TODO: Random this
                X_train = np.append(X_train, np.array([[0] * 300]), axis=0)
            else:
                word_vectors = self.vectorize_text_glove(tokenized_text)
                text_vector = np.sum(word_vectors, 0) / len(word_vectors)
                X_train = np.append(X_train, np.array([text_vector]), axis=0)
        return X_train

    def get_X_dev_glove_vec_mean(self):
        dev_docs = self.get_dev_docs()
        preprocessor = DataPreprocessor()
        tokenized_texts = preprocessor.preprocess_data(dev_docs)
        X_dev = np.empty((0, 300), float)
        for tokenized_text in tokenized_texts:
            if len(tokenized_text) == 0:
                X_dev = np.append(X_dev, np.array([[0] * 300]), axis=0)
            else:
                word_vectors = self.vectorize_text_glove(tokenized_text)
                text_vector = np.sum(word_vectors, 0) / len(word_vectors)
                X_dev = np.append(X_dev, np.array([text_vector]), axis=0)
        return X_dev

    def get_X_test_glove_vec_mean(self):
        test_docs = self.get_test_docs()
        preprocessor = DataPreprocessor()
        tokenized_texts = preprocessor.preprocess_data(test_docs)
        X_test = np.empty((0, 300), float)
        for tokenized_text in tokenized_texts:
            if len(tokenized_text) == 0:
                X_test = np.append(X_test, np.array([[0] * 300]), axis=0)
            else:
                word_vectors = self.vectorize_text_glove(tokenized_text)
                text_vector = np.sum(word_vectors, 0) / len(word_vectors)
                X_test = np.append(X_test, np.array([text_vector]), axis=0)
        return X_test

    def get_y_train(self):
        y_train = []
        for file_name in self.train_file_names:
            file_labels = self.get_label_text_from_file(file_name)[0]
            y_train.extend(file_labels)
        return y_train

    def get_y_dev(self):
        y_dev = []
        with open('data/dev.json') as file:
            data_json = json.load(file)
            for key in data_json.keys():
                y_dev.append(data_json[key]['label'])
        return y_dev