import re
import json
import nltk
import keras
import random
import numpy as np
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
from data_loader import dataLoader


class ClassicModelBuilder():

    def __init__(self):
        pass

    def process_logistic_regression(self, X_train, y_train, X_test):
        clf = LogisticRegression(random_state=0, class_weight={0: 0.6})
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def process_naive_bayes(self, X_train, y_train, X_test):
        nb = MultinomialNB(alpha=0.1, fit_prior=True)
        # nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        # nb_param_grid = {'alpha': np.arange(0.1, 1.1, 0.1)}
        # nb_search = GridSearchCV(nb, param_grid=nb_param_grid, scoring='recall')
        # nb_search.fit(X_train, y_train)
        # y_pred = nb_search.predict(X_test)
        return y_pred

    def process_kmeans(self):
        X = []
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(X)
        distances = kmeans.transform(X)
        sorted_idx = np.argsort(distances.ravel())[::-1][:5]

    def process_svm(self, X_train, y_train, X_test):
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        return y_test_pred

    def process_one_class_svm(self, X_train, y_train, X_test):
        clf = svm.OneClassSVM()
        clf.fit(X_train)
        y_test_pred = clf.predict(X_test)
        return y_test_pred

    def process_knn(self, X_train, y_train, X_test):
        # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train)
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        knn_clf.fit(X_train, y_train)
        return knn_clf.predict(X_test)
