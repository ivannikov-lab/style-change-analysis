from typing import Tuple, List
import os
import re
from .utils import EXTERNAL_DIR


try:
    with open(os.path.join(EXTERNAL_DIR, "abbrevs.txt"), 'r', encoding="utf8") as fin:
        ABBREVS = [abbrev_form for abbrev_form in fin.read().split()]

    LOWER_ABBREVS_SET = set(ABBREVS)
    UPPER_ABBREVS_SET = set([ABBREV.upper() for ABBREV in ABBREVS])

except Exception as e:
    print("Failed to read abbrevs.txt. Skipping abbreviation features.")

try:
    with open(os.path.join(EXTERNAL_DIR, "common_abbrevs.txt"), 'r', encoding="utf8") as fin:
        ABBREVS_FORMS = [abbrev_form.split() for abbrev_form in fin.readlines()]

    DOTED_WITHIN_ABBREVS = []
    for abbrev_forms in ABBREVS_FORMS:
        for abbrev_form in abbrev_forms:
            if re.search("․", abbrev_form[:-1]):
                DOTED_WITHIN_ABBREVS.append(abbrev_form)

    ABBREV_CONJ_SETS = []
    for abbrev_forms in ABBREVS_FORMS:
        FORMS = abbrev_forms[1:]
        abb_set = set()
        for form in FORMS:
            abb_set.add(form)
            abb_set.add(form + "-")
        ABBREV_CONJ_SETS.append(abb_set)

except Exception as e:
    print("Failed to read common_abbrevs.txt. Skipping common abbreviation features.")


def extract_abbreviation_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    features = []
    if feature_names is None:
        for feature in word_abbreviation_features.values():
            features.extend(feature(text, paragraph))
    else:
        for feature in feature_names:
            features.extend(word_abbreviation_features[feature](text, paragraph))
    return features


def abbrev_format(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_words = _without_dashes(paragraph)
    feature = []
    for abbrev_forms in ABBREVS_FORMS:
        FULL_FORM = abbrev_forms[0]
        SHORT_FORMS = abbrev_forms[1:]
        feature.append(float(FULL_FORM in text.lower() and any(abbrev in corr_words for abbrev in SHORT_FORMS)))
    return feature


def abbrev_declension_format(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_text_words = re.sub(r"—", "-", re.sub(r"\.", "․", text)).split()
    corr_words = _with_dashes(paragraph)
    feature = []
    for ABBREV_CONJ_SET in ABBREV_CONJ_SETS:
        sum = 0
        for abbrev in ABBREV_CONJ_SET:
            sum += (float(abbrev in corr_words and any(
                other_abbrev in corr_text_words for other_abbrev in ABBREV_CONJ_SET - {abbrev})))
        feature.append(float(sum > 0))
    return feature


def upper_cased_abbrev(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_text_words = re.sub(r"—.*|-.*", "", re.sub(r"\.|․", "", text)).split()
    corr_words = []
    for sent in paragraph:
        for word, lemma, pos, deprel in sent:
            word = re.sub(r"—.*|-.*", "", word)
            word = re.sub(r"\.|․", "", word)
            corr_words.append(word)
    feature = []
    for upper_abbrev in UPPER_ABBREVS_SET:
        feature.append(float(
            upper_abbrev in corr_words and any(lower_abbrev in corr_text_words for lower_abbrev in LOWER_ABBREVS_SET)))
    return feature


def lower_cased_abbrev(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_text_words = re.sub(r"—.*|-.*", "", re.sub(r"\.|․", "", text)).split()
    corr_words = []
    for sent in paragraph:
        for word, lemma, pos, deprel in sent:
            word = re.sub(r"—.*|-.*", "", word)
            word = re.sub(r"\.|․", "", word)
            corr_words.append(word)
    feature = []
    for lower_abbrev in LOWER_ABBREVS_SET:
        feature.append(float(
            lower_abbrev in corr_words and any(upper_abbrev in corr_text_words for upper_abbrev in UPPER_ABBREVS_SET)))
    return feature


def double_doted_abbrev(text: str, paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_words = _without_dashes(paragraph)
    feature = []
    for doted_within_abbrev in DOTED_WITHIN_ABBREVS:
        feature.append(float(doted_within_abbrev in corr_words))
    return feature


# corrects dot type to "․" and — to "-",
# adds wrong separated "-" and "․-"es,
# makes words lower cased
def _with_dashes(paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_words = []
    words = []
    for sent in paragraph:
        for word, pos, lemma, deprel in sent:
            words.append(word)
            word = re.sub(r"\.", "․", word)
            word = re.sub(r"—.*|-.*", "-", word)
            word = word.lower()
            if word == "․" or word.startswith("․-"):
                char = "․" if word == "․" else "․-"
                if len(corr_words):
                    corr_words[-1] += char
                else:
                    corr_words.append(char)
            if word.startswith("-"):
                if len(corr_words):
                    corr_words[-1] += "-"
                else:
                    corr_words.append("-")
            else:
                corr_words.append(word)
    return corr_words


# corrects dot type to "․" and dashes to "-",
# deletes everything after "-" with "-" itself,
# makes words lower cased
def _without_dashes(paragraph: List[List[Tuple[str, str, str, str]]]):
    corr_words = []
    for sent in paragraph:
        for word, lemma, pos, deprel in sent:
            word = re.sub(r"\.", "․", word)
            word = re.sub(r"—.*|-.*", "", word)
            word = word.lower()
            if word == "․" and corr_words:
                corr_words[-1] = corr_words[-1] + "․"
            else:
                corr_words.append(word)
    return corr_words


word_abbreviation_features = {
    'abbreviation_form': abbrev_format,
    'abbreviation_declension_format': abbrev_declension_format,
    'upper_cased_abbreviation': upper_cased_abbrev,
    'lower_cased_abbreviation': lower_cased_abbrev,
    'double_doted_abbreviation': double_doted_abbrev
}
