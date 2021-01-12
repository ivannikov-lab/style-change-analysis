from typing import Tuple, List
from collections import defaultdict, Counter
import re
import os
from .utils import EXTERNAL_DIR, nlp_stanza

dict_of_freqs = {}
try:
    with open(os.path.join(EXTERNAL_DIR, "words_with_freqs.txt"), 'r', encoding="utf8") as f:
        for line in f:
            key, val = line.split()
            dict_of_freqs[key] = int(val)
except Exception as e:
    print("Failed to read words_with_freqs.txt. Skipping word frequency features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "jargon_words.txt"), 'r', encoding="utf8") as f:
        jargon_words = f.read().splitlines()
except Exception as e:
    print("Failed to read jargon_words.txt. Skipping jargon features.")


def _get_list_of_uncommon_words():
    res = defaultdict(list)
    for key, val in sorted(dict_of_freqs.items()):
        res[val].append(key)
    return res[1]


list_of_uncommon_words = _get_list_of_uncommon_words()


def extract_word_general_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    avg_word_len_of_text = _get_average_word_len(text)
    features = []
    if feature_names is None:
        for feature in word_general_features.values():
            features.extend(feature(avg_word_len_of_text, paragraph))
    else:
        for feature in feature_names:
            features.extend(word_general_features[feature](avg_word_len_of_text, paragraph))
    return features


def _get_average_word_len(text: str):
    # finding average word length by symbols of text
    doc = nlp_stanza(text)
    words_count = 0
    all_words_len = 0
    for sentence in doc.sentences:
        for word in sentence.words:
            words_count += 1
            all_words_len += len(word.text)
    return all_words_len / words_count


def long_word_occurrence(avg_word_len_of_text, paragraph: List[List[Tuple[str, str, str, str]]]):
    paragraph_word_lens = [len(item[0]) for sent in paragraph for item in sent]
    occurrence = any(word_len > avg_word_len_of_text for word_len in paragraph_word_lens)
    return [float(occurrence)]


def long_words_freq(avg_word_len_of_text, paragraph: List[List[Tuple[str, str, str, str]]]):
    paragraph_word_lens = [len(item[0]) for sent in paragraph for item in sent]
    matches_count = sum(word_len > avg_word_len_of_text for word_len in paragraph_word_lens)
    return [float(matches_count / len(paragraph_word_lens))]


def find_not_armenian_letters(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words = [word[0] for sentence in paragraph for word in sentence]
    for char in "".join(words):
        if char.isalpha() and re.search("[^\u0561-\u0587\u0531-\u0556]", char) is not None:
            return [1.0]
    return [0.0]


def contains_jargon(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    lemmas = [word[1] for sentence in paragraph for word in sentence]
    for jargon_word in jargon_words:
        if jargon_word in lemmas:
            return [1.0]
    return [0.0]


def uncommon_words(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    lemmas = [word[1] for sentence in paragraph for word in sentence]
    for uncommon_word in list_of_uncommon_words:
        if uncommon_word in lemmas:
            return [1.0]
    return [0.0]


def uncommon_words_freq(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    lemmas = [word[1] for sentence in paragraph for word in sentence]
    c = Counter(lemmas)
    count = 0.0
    for word in list_of_uncommon_words:
        count += c[word]
    return [float(count / len(lemmas))]


def average_freq_of_words(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    lemmas = [word[1] for sentence in paragraph for word in sentence]
    return [sum([dict_of_freqs.get(word, 0) / len(lemmas) for word in lemmas])]


word_general_features = {
    'long_words_occurrence': long_word_occurrence,
    'long_words_frequency': long_words_freq,
    'uncommon_words_occurrence': uncommon_words,
    'uncommon_words_frequency': uncommon_words_freq,
    'words_average_frequency': average_freq_of_words,
    'not_armenian_words': find_not_armenian_letters,
    'not_formal_words': contains_jargon
}
