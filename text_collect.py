import os
import urllib
from urllib.request import urlopen
import bs4
from bs4 import BeautifulSoup
import html2text
import requests
import json
import pandas as pd
import nltk


def get_text_from_local_html():
    path = 'external_data'
    dirs = os.listdir(path)
    h = html2text.HTML2Text()
    h.ignore_links = True
    external_docs = {}
    counter = 0
    for folder in dirs:
        wiki_file = '{}/{}/{}'.format(path, folder, 'wiki.html')
        try:
            with open(wiki_file) as file:
                html = file.read()
                whole_text = h.handle(html)
                length = len(whole_text)
                text = whole_text[int(length*(1/4)):int(length*(1/4))+1000]
                print(text)
                doc_key = 'train-{}'.format(counter)
                external_docs[doc_key] = {}
                external_docs[doc_key]['text'] = text
                external_docs[doc_key]['label'] = 0
                counter += 1
                print('-----------------------------')
        except:
            pass

    with open('train-external-wiki.json', 'a') as file:
        json.dump(external_docs, file)


def get_text_from_html():
    h = html2text.HTML2Text()
    h.ignore_links = True
    with open('cnn_sources.txt') as file:
        for line in file:
            url = line
            res = requests.get(url)
            html_page = res.content
            soup = BeautifulSoup(html_page, 'html.parser')
            text = soup.get_text()
            print(h.handle(text))
            text = soup.find_all(text=True)
            # print(set([t.parent.name for t in text]))
            # output = ''
            # blacklist = [
            #     '[document]',
            #     'noscript',
            #     'header',
            #     'html',
            #     'meta',
            #     'head',
            #     'input',
            #     'script',
            #     # there may be more elements you don't want, such as "style", etc.
            # ]
            # for t in text:
            #     if t.parent.name not in blacklist:
            #         output += '{} '.format(t)
            # print(output)
            break


def get_text_from_csv():
    data = pd.read_csv('True.csv')
    wolrdnews = data[data['subject'] == 'worldnews']
    docs = {}
    counter = 0
    for text in wolrdnews['text']:
        doc_key = 'train-{}'.format(counter)
        docs[doc_key] = {}
        docs[doc_key]['text'] = text
        docs[doc_key]['label'] = 0
        counter += 1
    with open('train-external-news.json', 'a') as file:
        json.dump(docs, file)


def get_text_from_txt():
    path = 'bbc'
    dirs = os.listdir(path)
    docs = {}
    counter = 0
    for folder in dirs:
        try:
            topic_folder = '{}/{}'.format(path, folder)
            topic_files = os.listdir(topic_folder)
            for topic_file in topic_files:
                file_path = '{}/{}/{}'.format(path, folder, topic_file)
                with open(file_path) as file:
                    for index, line in enumerate(file):
                        if index == 6:
                            doc_key = 'train-{}'.format(counter)
                            docs[doc_key] = {}
                            docs[doc_key]['text'] = line
                            docs[doc_key]['label'] = 0
                            counter += 1
        except:
            pass

    with open('train-external-bbc-2.json', 'a') as file:
        json.dump(docs, file)

def collect_truth(input_file, output_file):
    counter = 0
    docs = {}
    with open(input_file) as file:
        for index, line in enumerate(file):
            if line.strip():
                doc_key = 'train-{}'.format(counter)
                docs[doc_key] = {}
                docs[doc_key]['text'] = line
                docs[doc_key]['label'] = 0
                counter += 1
    with open(output_file, 'w') as file:
        json.dump(docs, file)


def collect_misinfo(input_file, output_file):
    counter = 0
    docs = {}
    with open(input_file, encoding="utf-8") as file:
        for index, line in enumerate(file):
            if line.strip():
                doc_key = 'train-{}'.format(counter)
                docs[doc_key] = {}
                docs[doc_key]['text'] = line
                docs[doc_key]['label'] = 1
                counter += 1
    with open(output_file, 'w', encoding="utf-8") as file:
        json.dump(docs, file)


if __name__ == '__main__':
    # get_text_from_local_html()
    # get_text_from_html()
    # get_text_from_csv()
    # get_text_from_txt()

    input_file_truth = 'train-external-skeptical-truth.txt'
    output_file_truth = 'train-external-skeptical-truth.json'
    collect_truth(input_file_truth, output_file_truth)

    input_file_misinfo = 'train-external-skeptical-misinfo.txt'
    output_file_misinfo = 'train-external-skeptical-misinfo.json'
    collect_misinfo(input_file_misinfo, output_file_misinfo)






