import os
import re
import stanza
import spacy_udpipe


EXTERNAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_data')
nlp_udpipe = spacy_udpipe.load(lang="hy")
nlp_stanza = stanza.Pipeline(use_gpu=False, lang='hy',  processors='tokenize, mwt, pos, lemma, depparse')


def lemmatizer(text: str):
    doc = nlp_stanza(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]


def pos_tagger(text: str):
    doc = nlp_stanza(text)
    return [word.pos for sentence in doc.sentences for word in sentence.words]


def word_tokenize(text: str, remove_punctuation=False):
    text = remove_punct(text) if remove_punctuation else text
    doc = nlp_udpipe(text)
    return [word.text for word in doc]


def letter_tokenize(text: str):
    return list(re.sub(r'[^\u0561-\u0587\u0531-\u0556]', '', text))


def letters_and_numbers(text: str):
    return list(re.sub(r'[^\d\u0561-\u0587\u0531-\u0556]', '', text))


def remove_punct(text: str):
    return re.sub(r'[^\d\s\u0561-\u0587\u0531-\u0556]', ' ', text)


def remove_non_letters(text: str):
    return re.sub(r'[^\s\u0561-\u0587\u0531-\u0556]', ' ', text)


def sentence_tokenize(text: str):
    doc = nlp_udpipe(text)
    return [x.string for x in list(doc.sents)]
