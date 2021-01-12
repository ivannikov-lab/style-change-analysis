from typing import List
from .utils import nlp_stanza
from feature_extraction import extract_char_punct_features, extract_char_general_features, \
    extract_ngram_features, extract_abbreviation_features, extract_number_features, extract_word_general_features,\
    extract_sent_morphological_features, extract_sent_syntactic_features, extract_sent_general_features,\
    extract_readability_features

feature_extractors = {
    "char_punct": extract_char_punct_features,
    "char_other": extract_char_general_features,
    "word_abbrev": extract_abbreviation_features,
    "word_num": extract_number_features,
    "word_ngrams": extract_ngram_features,
    "word_others": extract_word_general_features,
    "sent_morph": extract_sent_morphological_features,
    "sent_syntax": extract_sent_syntactic_features,
    "sent_other": extract_sent_general_features,
    "p_readiblity": extract_readability_features
}


def compute_feature_vectors(text: str, paragraphs: List[str], feature_names: List[str]):
    feat_vectors = []
    for par in paragraphs:
        par_feat_vec = []
        par_doc = nlp_stanza(par)
        for extract_name in feature_names:
            par_feat_vec.extend(feature_extractors[extract_name](text,
                                                                 [[(w.text, w.lemma, w.pos, w.deprel)
                                                                  for w in sent.words] for sent in par_doc.sentences]))
        feat_vectors.append(par_feat_vec)
    return feat_vectors
