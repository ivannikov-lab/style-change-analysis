from typing import Tuple, List
import os
from nltk import ngrams
from .utils import EXTERNAL_DIR

try:
    with open(os.path.join(EXTERNAL_DIR, "pos_labels.txt"), 'r', encoding="utf8") as fin:
        pos_unigrams = [pos for pos in fin.read().split()]
except Exception as e:
    print("Failed to read pos_labels.txt. Skipping pos label features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "common_pos_bigrams.txt"), 'r', encoding="utf8") as fin:
        possible_pos_bigrams = [tuple(poses.split()) for poses in fin.read().split("\n")]
except Exception as e:
    print("Failed to read common_pos_bigrams.txt. Skipping pos bigram features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "common_pos_trigrams.txt"), 'r', encoding="utf8") as fin:
        possible_pos_trigrams = [tuple(poses.split()) for poses in fin.read().split("\n")]
except Exception as e:
    print("Failed to read common_pos_trigrams.txt. Skipping pos trigram features.")


def extract_sent_morphological_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    paragraph_pos_unigrams = [item[2] for sentence in paragraph for item in sentence]
    if feature_names is None:
        for feature in sent_morphological_features.values():
            features.extend(feature(text, paragraph_pos_unigrams))
    else:
        for feature in feature_names:
            features.extend(sent_morphological_features[feature](text, paragraph_pos_unigrams))
    return features


def pos_unigram_freq(text: str, paragraph_pos_unigrams: list):
    feature = [float(paragraph_pos_unigrams.count(pos) / len(pos_unigrams))
               for pos in pos_unigrams]
    return feature


def pos_bigram_freq(text: str, paragraph_pos_unigrams: list):
    paragraph_pos_bigrams = list(ngrams(paragraph_pos_unigrams, 2))
    feature = [float(paragraph_pos_bigrams.count(pos) / (len(possible_pos_bigrams) - 1))
               if len(possible_pos_bigrams)-1 != 0 else [0.0]
               for pos in possible_pos_bigrams]
    return feature


def pos_trigram_freq(text: str, paragraph_pos_unigrams: list):
    paragraph_pos_trigrams = list(ngrams(paragraph_pos_unigrams, 3))
    feature = [float(paragraph_pos_trigrams.count(pos) / (len(possible_pos_trigrams) - 2))
               if len(possible_pos_trigrams)-2 != 0 else [0.0]
               for pos in possible_pos_trigrams]
    return feature


def pos_bigram_occurrence(text: str, paragraph_pos_unigrams: list):
    paragraph_pos_bigrams = list(ngrams(paragraph_pos_unigrams, 2))
    feature = [(float(pos in paragraph_pos_bigrams))for pos in possible_pos_bigrams]
    return feature


def pos_trigram_occurrence(text: str, paragraph_pos_unigrams: list):
    paragraph_pos_trigrams = list(ngrams(paragraph_pos_unigrams, 3))
    feature = [(float(pos in paragraph_pos_trigrams))for pos in possible_pos_trigrams]
    return feature


sent_morphological_features = {
    "pos_bigram_occurrence": pos_bigram_occurrence,
    "pos_trigram_occurrence": pos_trigram_occurrence,
    "pos_unigram_frequency": pos_unigram_freq,
    "pos_bigram_frequency": pos_bigram_freq,
    "pos_trigram_frequency": pos_trigram_freq
}
