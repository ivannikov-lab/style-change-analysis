from typing import Tuple, List
from .utils import nlp_stanza


def extract_sent_general_features(text: str, paragraph: List[List[Tuple[str, str, str, str]]], feature_names=None):
    doc = nlp_stanza(text)
    features = []
    if feature_names is None:
        for feature in sent_general_features.values():
            features.extend(feature(doc, paragraph))
    else:
        for feature in feature_names:
            features.extend(sent_general_features[feature](text, paragraph))
    return features


def _len_by_symbols(sentence: List):
    return sum(len(item[0]) for item in sentence)


def avg_sent_len_by_symbols(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    return [float(sum(sent_len for sent_len in sent_lens_by_symb) / len(paragraph))]


def avg_sent_len_by_words(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    all_words_count = sum(len(sent) for sent in paragraph)
    return [float(all_words_count / len(paragraph))]


def max_sent_len_by_symbols(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    return [float((max(sent_lens_by_symb)))]


def max_sent_len_by_words(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_lens_by_words = [len(sent) for sent in paragraph]
    return [float(max(sent_lens_by_words))]


def min_sent_len_by_symbols(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    return [float((min(sent_lens_by_symb)))]


def min_sent_len_by_words(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    sent_lens_by_words = [len(sent) for sent in paragraph]
    return [float(min(sent_lens_by_words))]


def long_sent_by_symbols_occurrence(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by symbols of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len_of_text = avg_sent_len_by_symbols(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    occurrence = any(sent_len >= avg_sent_len_of_text for sent_len in sent_lens_by_symb)
    return [float(occurrence)]


def long_sent_by_words_occurrence(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by symbols of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len_of_text = avg_sent_len_by_words(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_words = [len(sent) for sent in paragraph]
    occurrence = any(sent_len > avg_sent_len_of_text for sent_len in sent_lens_by_words)
    return [float(occurrence)]


def long_sents_by_symbols_freq(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by words of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len = avg_sent_len_by_symbols(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    matches_count = sum(sent_len > avg_sent_len for sent_len in sent_lens_by_symb)
    return [float(matches_count / len(paragraph))]


def long_sents_by_words_freq(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by words of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len = avg_sent_len_by_words(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_words = [len(sent) for sent in paragraph]
    matches_count = sum(sent_len > avg_sent_len for sent_len in sent_lens_by_words)
    return [float(matches_count / len(paragraph))]


def short_sent_by_symbols_occurrence(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by symbols of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len_of_text = avg_sent_len_by_symbols(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    occurrence = any(sent_len <= avg_sent_len_of_text for sent_len in sent_lens_by_symb)
    return [float(occurrence)]


def short_sent_by_words_occurrence(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by symbols of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len_of_text = avg_sent_len_by_words(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_words = [len(sent) for sent in paragraph]
    occurrence = any(sent_len <= avg_sent_len_of_text for sent_len in sent_lens_by_words)
    return [float(occurrence)]


def short_sents_by_symbols_freq(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by words of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len = avg_sent_len_by_symbols(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_symb = [_len_by_symbols(sent) for sent in paragraph]
    matches_count = sum(sent_len <= avg_sent_len for sent_len in sent_lens_by_symb)
    return [float(matches_count / len(paragraph))]


def short_sents_by_words_freq(doc, paragraph: List[List[Tuple[str, str, str, str]]]):
    # finding average sentence length by words of text
    txt = []
    for sentence in doc.sentences:
        txt_sent = []
        for word in sentence.words:
            txt_sent.append((word.text, word.lemma, word.pos, word.deprel))
        txt.append(txt_sent)
    avg_sent_len = avg_sent_len_by_words(doc, txt)[0]
    # finding short sentences frequency in paragraph
    sent_lens_by_words = [len(sent) for sent in paragraph]
    matches_count = sum(sent_len > avg_sent_len for sent_len in sent_lens_by_words)
    return [float(matches_count / len(paragraph))]


sent_general_features = {
    "_words_short_sent_occurrence": short_sent_by_words_occurrence,
    "_words_long_sent_occurrence": long_sents_by_words_freq,
    "_words_short_sents_frequency": short_sents_by_words_freq,
    "_words_long_sents_frequency": long_sents_by_words_freq,
    "_words_min_sent_length": min_sent_len_by_words,
    "_words_avg_sent_length": avg_sent_len_by_words,
    "_words_max_sent_length": max_sent_len_by_words,

    "_symbols_short_sent_occurrence": short_sent_by_symbols_occurrence,
    "_symbols_long_sent_occurrence": long_sent_by_symbols_occurrence,
    "_symbols_short_sents_frequency": short_sents_by_symbols_freq,
    "_symbols_long_sents_frequency": long_sents_by_symbols_freq,
    "_symbols_min_sent_length": min_sent_len_by_symbols,
    "_symbols_avg_sent_length": avg_sent_len_by_symbols,
    "_symbols_max_sent_length": max_sent_len_by_symbols
}