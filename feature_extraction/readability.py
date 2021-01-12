from typing import Tuple, List
import math
import os
from .utils import letter_tokenize, letters_and_numbers, sentence_tokenize, EXTERNAL_DIR
from collections import Counter

try:
    with open(os.path.join(EXTERNAL_DIR, "syllables.txt"), 'r', encoding="utf8") as fin:
        SYLLABLES_LIST = fin.read().splitlines()
except Exception as e:
    print("Failed to read syllables.txt. Skipping readability features.")


def syllables_counter(text: str):
    c = Counter(text)
    return sum([c[syllable]for syllable in SYLLABLES_LIST])


def extract_readability_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in readability_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(readability_features[feature](text, paragraph))
    return features


def flesch_reading_ease(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words_count = sum([len(sentence) for sentence in paragraph])
    syllables_count = sum([syllables_counter(word[0]) for sentence in paragraph for word in sentence])
    sentences_count = len(paragraph)
    try:
        FSE = 78.39 + 2.6 * (words_count / sentences_count) - 32.3 * (syllables_count / words_count)
    except ZeroDivisionError:
        return 0.0
    return [round(FSE, 2)]


def smog_index(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    syllables_count = sum([syllables_counter(word[0]) for sentence in paragraph for word in sentence])
    sentences_count = len(paragraph)
    try:
        smog = 0.6 * math.sqrt(syllables_count / sentences_count) + 9.0
    except ZeroDivisionError:
        return 0.0
    return [round(smog, 2)]


def flesch_kincaid_grade(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words_count = sum([len(sentence) for sentence in paragraph])
    syllables_count = sum([syllables_counter(word[0]) for sentence in paragraph for word in sentence])
    sentences_count = len(paragraph)
    try:
        FK = -0.33 * (words_count / sentences_count) + 6.42 * (syllables_count / words_count) + 4.7
    except ZeroDivisionError:
        return 0.0
    return [round(FK, 2)]


def coleman_liau_index(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    letters_count = sum([len(letter_tokenize(word[0])) for sentence in paragraph for word in sentence])
    words_count = sum([len(sentence) for sentence in paragraph])
    sentences_count = len(paragraph)
    try:
        CL = 1.2 * (letters_count / words_count) + 62.65 * (sentences_count / words_count) + 0.662
    except ZeroDivisionError:
        return 0.0
    return [round(CL, 2)]


def automated_readability_index(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    letters_and_nums_count = sum([len(letters_and_numbers(word[0])) for sentence in paragraph for word in sentence])
    words_count = sum([len(sentence) for sentence in paragraph])
    sentences_count = len(paragraph)
    try:
        AT = 3.062 * (letters_and_nums_count / words_count) - 0.049 * (words_count / sentences_count) + 0.078
    except ZeroDivisionError:
        return 0.0
    return [round(AT, 2)]


def dale_chall_readability_score(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words = [word[0] for sentence in paragraph for word in sentence]
    words_count = len(words)
    c = 0.0
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    count = words_count - c
    try:
        per = count / (words_count * 100)
    except ZeroDivisionError:
        return 0.0
    difficult_words = 100 - per
    score = ((0.1579 * difficult_words) + (0.0496 * words_count / len(paragraph)))
    if difficult_words > 5:
        score += 3.6365
    return [round(score, 2)]


def linsear_write_formula(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words = [word[0] for sentence in paragraph for word in sentence][:100]
    sentences_count = len(sentence_tokenize(' '.join(words)))
    c1 = 0.0
    c3 = 0.0
    for word in words:
        if syllables_counter(word) < 3:
            c1 = c1 + 1
        else:
            c3 = c3 + 1
    try:
        lin = (c1 + c3) / sentences_count
    except ZeroDivisionError:
        return 0.0
    return [round(lin, 2)]


def difficult_words(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    c = 0.0
    words = [word[0] for sentence in paragraph for word in sentence]
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    return [c]


def gunning_fog(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    words = [word[0] for sentence in paragraph for word in sentence]
    words_count = len(words)
    sentences_count = len(paragraph)
    c = 0.0
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    try:
        GF = 0.4 * (words_count / sentences_count + 100 * (c / words_count))
    except ZeroDivisionError:
        return 0.0
    return [round(GF, 2)]


readability_features = {
    "flesch_reading_ease": flesch_reading_ease,
    "smog_grade": smog_index,
    "flesch_kincaid_grade": flesch_kincaid_grade,
    "coleman_liau_index": coleman_liau_index,
    "automated_readability_index": automated_readability_index,
    "dale_chall_readability_score": dale_chall_readability_score,
    "difficult_words": difficult_words,
    "linsear_write_formula": linsear_write_formula,
    "gunning_fog": gunning_fog
}
