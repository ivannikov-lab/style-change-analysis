from typing import Tuple, List
import os
import sys
from .utils import remove_non_letters, EXTERNAL_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


UNIGRAM_IDF_THRESHOLD = 4.0
BIGRAM_IDF_THRESHOLD = 5.0
TRIGRAM_IDF_THRESHOLD = 6.0


def load_ngrams(n: int):
    if n not in [1, 2, 3]:
        raise ModuleNotFoundError("Incorrect number, choose on of these: 1,2,3")
    path = os.path.join(EXTERNAL_DIR, ["unigrams_with_idf.txt", "bigrams_with_idf.txt", "trigrams_with_idf.txt"][n-1])
    dict_of_freqs = {}
    with open(path, "r") as f:
        for line in f:
            res = line.split()
            dict_of_freqs[' '.join(res[:-1])] = float(res[-1])
    return dict_of_freqs


try:
    unigrams_with_freqs = load_ngrams(1)
    bigrams_with_freqs = load_ngrams(2)
    trigrams_with_freqs = load_ngrams(3)
except Exception as e:
    print("Failed to read ngram files. Skipping word ngram features.")


def extract_ngram_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    par_text = ' '.join([token[1] for sentence in paragraph for token in sentence])
    par_text = remove_non_letters(par_text)
    par_text = par_text.split(' ')
    features = []
    if feature_names is None:
        for feature in n_grams_features.values():
            features.extend(feature(text, par_text))
    else:
        for feature in feature_names:
            features.extend(n_grams_features[feature](text, par_text))
    return features


def save_unigrams_with_idf(documents: List[str]):
    documents = [remove_non_letters(doc) for doc in documents]
    vectorizer = TfidfVectorizer(smooth_idf=False, use_idf=True, stop_words=stopwords.words('armenian'))
    vectorizer.fit_transform(documents)
    with open(os.path.join(EXTERNAL_DIR, "unigrams_with_idf.txt"), 'w') as f:
        for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_):
            f.write("{} {}\n".format(word, idf))


def save_bigrams_with_idf(documents: List[str]):
    documents = [remove_non_letters(doc) for doc in documents]
    vectorizer = TfidfVectorizer(smooth_idf=False, use_idf=True, ngram_range=(2, 2), stop_words=stopwords.words('armenian'))
    vectorizer.fit_transform(documents)
    with open(os.path.join(EXTERNAL_DIR, "bigrams_with_idf.txt"), 'w') as f:
        for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_):
            f.write("{} {}\n".format(word, idf))


def save_trigrams_with_idf(documents: List[str]):
    documents = [remove_non_letters(doc) for doc in documents]
    vectorizer = TfidfVectorizer(smooth_idf=False, use_idf=True, ngram_range=(3, 3), stop_words=stopwords.words('armenian'))
    vectorizer.fit_transform(documents)
    with open(os.path.join(EXTERNAL_DIR, "trigrams_with_idf.txt"), 'w') as f:
        for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_):
            f.write("{} {}\n".format(word, idf))


def get_bigrams(text: list):
    return [' '.join(bigram) for bigram in zip(text[:-1], text[1:])]


def get_trigrams(text: list):
    return [' '.join(trigram) for trigram in zip(text[:-1], text[1:], text[2:])]


def find_unigrams_with_low_idf(text: str, paragraph: list):
    unigrams = paragraph
    for token in unigrams:
        if unigrams_with_freqs.get(token, sys.maxsize) < UNIGRAM_IDF_THRESHOLD:
            return [1.0]
    return [0.0]


def find_bigrams_with_low_idf(text: str, paragraph: list):
    bigrams = get_bigrams(paragraph)
    for bigram in bigrams:
        if bigrams_with_freqs.get(bigram, sys.maxsize) < BIGRAM_IDF_THRESHOLD:
            return [1.0]
    return [0.0]


def find_trigrams_with_low_idf(text: str, paragraph: list):
    trigrams = get_trigrams(paragraph)
    for trigram in trigrams:
        if trigrams_with_freqs.get(trigram, sys.maxsize) < BIGRAM_IDF_THRESHOLD:
            return [1.0]
    return [0.0]


def unigrams_with_low_idf(text: str, paragraph: list):
    unigrams = paragraph
    c = 0.0
    for token in unigrams:
        if unigrams_with_freqs.get(token, sys.maxsize) < UNIGRAM_IDF_THRESHOLD:
            c += 1
    return [c / len(unigrams)]


def bigrams_with_low_idf(text: str, paragraph: list):
    bigrams = get_bigrams(paragraph)
    c = 0.0
    for bigram in bigrams:
        if bigrams_with_freqs.get(bigram, sys.maxsize) < BIGRAM_IDF_THRESHOLD:
            c += 1
    return [c / (len(bigrams) - 1)if len(bigrams)-1 != 0 else 0.0]


def trigrams_with_low_idf(text: str, paragraph: list):
    trigrams = get_trigrams(paragraph)
    c = 0.0
    for bigram in trigrams:
        if trigrams_with_freqs.get(bigram, sys.maxsize) < BIGRAM_IDF_THRESHOLD:
            c += 1
    return [c / (len(trigrams) - 2)if len(trigrams)-2 != 0 else 0.0]


n_grams_features = {
    "low_idf_unigrams_occurrence": find_unigrams_with_low_idf,
    "low_idf_bigrams_occurrence": find_bigrams_with_low_idf,
    "low_idf_trigrams_occurrence": find_trigrams_with_low_idf,
    "low_idf_unigrams_rate": unigrams_with_low_idf,
    "low_idf_bigrams_rate": bigrams_with_low_idf,
    "low_idf_trigrams_rate": trigrams_with_low_idf
}
