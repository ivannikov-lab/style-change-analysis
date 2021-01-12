from typing import Tuple, List
import os
from .utils import EXTERNAL_DIR

deprels = {}
try:
    with open(os.path.join(EXTERNAL_DIR, "dependency_labels.txt"), 'r', encoding="utf8") as labels_src:
        labels = labels_src.read().splitlines()
        for i, label in enumerate(labels):
            deprels[label] = i
except Exception as e:
    print("Failed to read dependency_labels.txt. Skipping syntactic features.")


def extract_sent_syntactic_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in sent_syntactic_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(sent_syntactic_features[feature](text, paragraph))
    return features


def extract_dependency_labels(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    features = [0.0] * len(deprels)
    for sent in paragraph:
        for token in sent:
            features[deprels[token[3]]] = 1.0
    return features


def extract_dependency_labels_rate(text: str, paragraph: List[List[Tuple[str, str, str, str]]], ngrams=(1, 2)):
    features = [0.0] * len(deprels)
    sent_count = len(paragraph)
    for sent in paragraph:
        for token in sent:
            features[deprels[token[3]]] += 1.0 / sent_count
    return features


def _get_complex_sentence_rate(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_count = 0
    complex_sent_labels = {'conj', 'ccomp', 'advcl', 'csubj', 'csubj:pass', 'acl'}
    for sent in paragraph:
        dep_labels = [token[3] for token in sent]
        if any(label in complex_sent_labels for label in dep_labels):
            sent_count += 1
    return [sent_count / len(paragraph)]


def extract_sentence_complexity_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    complex_rate = _get_complex_sentence_rate(text, paragraph)[0]
    return [complex_rate, float(complex_rate < 1.0), float(complex_rate > 0.0)]


sent_syntactic_features = {
    "dep_labels": extract_dependency_labels,
    "dep_labels_rate": extract_dependency_labels_rate,
    "sentence_complexity": extract_sentence_complexity_features
}
