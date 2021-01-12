from typing import Tuple, List
import os
from .utils import EXTERNAL_DIR

try:
    with open(os.path.join(EXTERNAL_DIR, "suffixes.txt"), 'r', encoding="utf8") as fin:
        SUFFIXES_LIST = fin.read().splitlines()
except Exception as e:
    print("Failed to read suffixes.txt. Skipping suffix features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "prefixes.txt"), 'r', encoding="utf8") as fin:
        PREFIXES_LIST = fin.read().splitlines()
except Exception as e:
    print("Failed to read prefixes.txt. Skipping prefix features.")


def extract_char_general_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in char_general_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(char_general_features[feature](text, paragraph))
    return features


def suffixes_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    for suffix in SUFFIXES_LIST:
        matches_count = sum(word.endswith(suffix) for word in paragraph_words)
        feature.append(matches_count)
    return [float(feat / len(paragraph_words)) for feat in feature]


def prefixes_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    for prefix in PREFIXES_LIST:
        matches_count = sum(word.startswith(prefix) for word in paragraph_words)
        feature.append(matches_count)
    return [float(feat / len(paragraph_words)) for feat in feature]


def suffixes_occurrence(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    for suffix in SUFFIXES_LIST:
        occurrence = any(word.endswith(suffix) for word in paragraph_words)
        feature.append(float(occurrence))
    return feature


def prefixes_occurrence(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    for prefix in PREFIXES_LIST:
        occurrence = any(word.startswith(prefix) for word in paragraph_words)
        feature.append(float(occurrence))
    return feature


char_general_features = {
    "suffixes_occurrence": suffixes_occurrence,
    "prefixes_occurrence": prefixes_occurrence,
    "suffixes_frequency": suffixes_freq,
    "prefixes_frequency": prefixes_freq
}
