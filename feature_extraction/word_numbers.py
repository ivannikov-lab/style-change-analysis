from typing import Tuple, List
import re
import os
from .utils import EXTERNAL_DIR

try:
    with open(os.path.join(EXTERNAL_DIR, "cardinal_numbers.txt"), 'r', encoding="utf8") as f:
        cardinal_nums = f.read().replace('\n', '|')[:-1]
except Exception as e:
    print("Failed to read cardinal_numbers.txt. Skipping punctuation features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "ordinal_numbers.txt"), 'r', encoding="utf8") as f:
        ordinal_nums = f.read().replace('\n', '|')[:-1]
except Exception as e:
    print("Failed to read ordinal_numbers.txt. Skipping ordinal number features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "months.txt"), 'r', encoding="utf8") as f:
        months = f.read().replace('\n', '|')[:-1]
except Exception as e:
    print("Failed to read months.txt. Skipping date format features.")


def extract_number_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in word_numbers_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(word_numbers_features[feature](text, paragraph))
    return features


def cardinal_numbers(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    tokens = [token for sentence in paragraph for token in sentence]
    forms = [0.0, 0.0]
    for token in tokens:
        if not forms[0] and re.search(cardinal_nums, token[1]) and not token[1].endswith('շաբթ'):
            forms[0] += 1
        if not forms[1] and token[2] == 'NUM':
            forms[1] += 1
        if forms == [1.0, 1.0]:
            break
    return forms


def ordinal_numbers(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    tokens = [token for sentence in paragraph for token in sentence]
    forms = [0.0] * 5
    for token in tokens:
        if not forms[0] and re.search(ordinal_nums, token[1]):
            forms[0] += 1
        if not forms[1] and re.search('i|x|v', token[0]):
            forms[1] += 1
        if not forms[2] and re.search('I|X|V', token[0]):
            forms[2] += 1
        if not forms[3] and re.search('\d-րդ', token[0]):
            forms[3] += 1
        if not forms[4] and re.search('\dրդ', token[0]):
            forms[4] += 1
        if forms == [1.0] * 5:
            break
    return forms


def date_format(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    forms = [0.0] * 9
    par_text = ' '.join([word[0] for sentence in paragraph for word in sentence])
    if re.search(r"(0*[1-30-9]){1,2}\\(0*[1-90-2]){1,2}\\\d{4}", par_text):
        forms[0] += 1
    elif re.search(r"(0*[1-30-9]){1,2}\\(0*[1-90-2]){1,2}\\\d{2}", par_text):
        forms[1] += 1
    if re.search(r"(0*[1-30-9]){1,2}\/(0*[1-90-2]){1,2}\/\d{4}", par_text):
        forms[2] += 1
    elif re.search(r"(0*[1-30-9]){1,2}\/(0*[1-90-2]){1,2}\/\d{2}", par_text):
        forms[3] += 1
    if re.search(r"(0*[1-30-9]){1,2}(\.|\․)(0*[1-90-2]){1,2}(\.|\․)\d{4}", par_text):
        forms[4] += 1
    elif re.search(r"(0*[1-30-9]){1,2}(\.|\․)(0*[1-90-2]){1,2}(\.|\․)\d{2}", par_text):
        forms[5] += 1
    if re.search(r"(0*[1-30-9]){1,2}(-|֊)(0*[1-90-2]){1,2}-\d{4}", par_text):
        forms[6] += 1
    elif re.search(r"(0*[1-30-9]){1,2}(-|֊)(0*[1-90-2]){1,2}-\d{2}", par_text):
        forms[7] += 1
    if re.search(r"(0*[1-30-9]){1,2},*\s*(" + months + ")(\u056B|\u053B)?\s*,*\s*\d{4}", par_text):
        forms[8] += 1
    return forms


word_numbers_features = {
    "cardinal_numbers": cardinal_numbers,
    "ordinal_numbers": ordinal_numbers,
    "date_format": date_format
}
