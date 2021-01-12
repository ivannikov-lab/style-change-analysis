from typing import Tuple, List
import os
from .utils import EXTERNAL_DIR

try:
    punct_path = os.path.join(EXTERNAL_DIR, "puncts.txt")
    with open(punct_path, 'r', encoding="utf8") as fin:
        PUNCT_LIST = fin.read().splitlines()
except Exception as e:
    print("Failed to read puncts.txt. Skipping punctuation features.")


def extract_char_punct_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in char_punctuation_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(char_punctuation_features[feature](text, paragraph))
    return features


def puncts_occurrence(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            occurrence = any(word.find(punctuation) != -1 for word in paragraph_words)
            feature.append(float(occurrence))
    else:
        feature = [float(any(word.find(punct) != -1 for word in paragraph_words))]
    return feature


def puncts_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            matches_count = sum(word.find(punctuation) != -1 for word in paragraph_words)
            feature.append(float(matches_count))
    else:
        feature = [float(sum(word.find(punct) != -1 for word in paragraph_words))]
    return [feat / len(paragraph_words) for feat in feature]


def before_spaced_puncts_occurrence(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    # Warning: if punctuation mark is separated as a single word-token, it won't be computed in feature
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            occurrence = any(word.startswith(punctuation) and word != punctuation for word in paragraph_words)
            feature.append(float(occurrence))
    else:
        feature = [float(any(word.startswith(punct) and word != punct for word in paragraph_words))]
    return feature


def before_spaced_puncts_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    # note if punctuation mark is separated as a single word-token, it won't be computed in feature
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            matches_count = sum(word.startswith(punctuation) and word != punctuation for word in paragraph_words)
            feature.append(float(matches_count))
    else:
        feature = [float(sum(word.startswith(punct) and word != punct for word in paragraph_words))]
    return [feat / len(paragraph_words) for feat in feature]


def after_spaced_puncts_occurrence(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    # Warning: if punctuation mark is separated as a single word-token, it won't be computed in feature
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            occurrence = any(word.endswith(punctuation) and word != punctuation for word in paragraph_words)
            feature.append(float(occurrence))
    else:
        feature = [float(any(word.endswith(punct) and word != punct for word in paragraph_words))]
    return feature


def after_spaced_puncts_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]], punct: str = None):
    # note if punctuation mark is separated as a single word-token, it won't be computed in feature
    feature = []
    paragraph_words = [item[0] for sentence in paragraph for item in sentence]
    if punct is None:
        for punctuation in PUNCT_LIST:
            matches_count = sum(word.endswith(punctuation) and word != punctuation for word in paragraph_words)
            feature.append(float(matches_count))
    else:
        feature = [float(sum(word.endswith(punct) and word != punct for word in paragraph_words))]
    return [feat / len(paragraph_words) for feat in feature]


char_punctuation_features = {
    "puncts_occurrence": puncts_occurrence,
    "before_spaced_puncts_occurrence": before_spaced_puncts_occurrence,
    "after_spaced_puncts_occurence": after_spaced_puncts_occurrence,
    "puncts_frequency": puncts_freq,
    "before_spaced_puncts_frequency": before_spaced_puncts_freq,
    "after_spaced_puncts_frequency": after_spaced_puncts_freq
}