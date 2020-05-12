import json
import numpy as np
import wget
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import dataLoader
from classic_models import ClassicModelBuilder
from squential_models import TextCNNBuilder


# Download Glove. Please note it takes 5.5GB storage.
# url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
# wget.download(url, 'glove.840B.300d.zip')


def load_glove_index():
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    print('Loading Glove pre-trained word vectors ...')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, 'r', encoding='UTF-8'))
    print('{} Glove word vectors loaded.'.format(len(embeddings_index)))
    return embeddings_index

def load_glove_vec_dict():
    print('Loading Glove pre-trained word vectors ...')
    vec_dict = {}
    with open('glove.840B.300d.txt', 'r', encoding='UTF-8') as glove_file:
        for line in glove_file:
            try:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                vec_dict[word] = embedding
            except:
                pass
    print('{} Glove word vectors loaded.'.format(len(vec_dict)))
    return vec_dict


def generate_dev_output(y_dev):
    with open('dev-baseline-r.json') as file:
        data_json = json.load(file)
        for index, key in enumerate(data_json.keys()):
            data_json[key]['label'] = int(y_dev[index])
    with open('dev-baseline-r.json', 'w') as file:
        json.dump(data_json, file)
    print('Development output generated')


def generate_output(y_test):
    with open('output/test-output.json') as file:
        data_json = json.load(file)
        for index, key in enumerate(data_json.keys()):
            data_json[key]['label'] = int(y_test[index])
    with open('output/test-output.json', 'w') as file:
        json.dump(data_json, file)
    print('Output Generated.')


if __name__ == '__main__':
    # vec_dict = load_glove_index()

    data_loader = dataLoader()
    X_train = data_loader.get_X_train_tfidf().toarray()
    X_dev = data_loader.get_X_dev_tfidf().toarray()
    X_test = data_loader.get_X_test_tfidf().toarray()
    y_train = data_loader.get_y_train()
    y_dev = data_loader.get_y_dev()
    # print('X_train shape: ', X_train.shape)
    # X_train_vec_mean = data_loader.get_X_train_glove_vec_mean()
    # X_dev_vec_mean = data_loader.get_X_dev_glove_vec_mean()
    # X_test_vec_mean = data_loader.get_X_test_glove_vec_mean()
    # print('X_train_vec_mean shape: ', X_train_vec_mean.shape)
    # X_train = np.concatenate((X_train, X_train_vec_mean), axis=1)
    # X_dev = np.concatenate((X_dev, X_dev_vec_mean), axis=1)
    # X_test = np.concatenate((X_test, X_test_vec_mean), axis=1)

    # cnn_builder = TextCNNBuilder()
    # y_test_pred = cnn_builder.process_CNN_word_vector()
    # y_test_pred = cnn_builder.process_cnn_embedding()

    model_builder = ClassicModelBuilder()
    y_dev_pred = model_builder.process_logistic_regression(X_train, y_train, X_dev)
    y_test_pred = model_builder.process_logistic_regression(X_train, y_train, X_test)

    print('Accuracy on development set: ', accuracy_score(y_dev, y_dev_pred))
    print('Precision on development set: ', precision_score(y_dev, y_dev_pred))
    print('F1-Score on development set: ', f1_score(y_dev, y_dev_pred))
    print('Recall on development set: ', recall_score(y_dev, y_dev_pred))
    print('Zero percentage on development set: ', sum([1 for y in y_dev_pred if y == 0]) / len(y_dev_pred))
    print('Zero percentage on test set: ', sum([1 for y in y_test_pred if y == 0]) / len(y_test_pred))

    generate_output(y_test_pred)

