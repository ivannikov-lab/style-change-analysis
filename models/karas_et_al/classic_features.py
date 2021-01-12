import os
from models.karas_et_al import external_data
from feature_extraction.utils import word_tokenize
karas_external_data_path = os.path.dirname(external_data.__file__)


def get_punct_list():
    return open(karas_external_data_path + '/' + 'puncts.txt', 'r', encoding='utf8').read().splitlines()


def get_stopwords_list():
    return open(karas_external_data_path + '/' + 'stopwords.txt', 'r', encoding='utf-8-sig').read().splitlines()


def punct_tokenize(text: str):
    punct_list = get_punct_list()
    punct_tokens = []
    for item in text:
        if item in punct_list:
            punct_tokens.append(item)
    return punct_tokens


def stopword_tokenize(text: str):
    stopwords_list = get_stopwords_list()
    word_tokens = word_tokenize(text)
    stopword_tokens = []
    for item in word_tokens:
        if item in stopwords_list:
            stopword_tokens.append(item)
    return stopword_tokens
