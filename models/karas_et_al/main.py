import operator
import scipy.sparse as sp
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extraction.utils import word_tokenize, pos_tagger
from models.karas_et_al.classic_features import punct_tokenize, stopword_tokenize
from models.utils.text_segmentation import get_start_indices


def find_style_change_starts_by_wilcoxon(features_data_frame, hyperparams):
    style_change_borders = [0]
    num_of_paragraphs = features_data_frame.shape[0]

    # doing Wilcoxon sign-rank test
    p_values = {}
    for i in range(num_of_paragraphs - 1):
        if features_data_frame.iloc[i].equals(features_data_frame.iloc[i + 1]) is False:
            stat, p_value = wilcoxon(features_data_frame.iloc[i], features_data_frame.iloc[i + 1],
                                     zero_method='pratt',
                                     alternative='two-sided')
            p_values[i + 1] = p_value
    sorted_p_values = sorted(p_values.items(), key=operator.itemgetter(1))

    # used hyper parameters
    allowed_portion = 0.3 if not hyperparams else hyperparams[0]
    alpha_value = 1 if not hyperparams else hyperparams[1]

    # defining % of suspicious parts
    S = int(allowed_portion * num_of_paragraphs)
    if allowed_portion != 1:
        S += 1

    # making a decision whether is a border
    for tpl in (sorted_p_values[:S]):
        if tpl[1] <= alpha_value:
            style_change_borders.append(tpl[0])
        else:
            break
    style_change_borders.sort()
    return style_change_borders if len(style_change_borders) != 1 else []


def get_classic_feature_vecs(paragraphs: list):
    # 1.word tfidf
    word_vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    word_vectors = word_vectorizer.fit_transform(paragraphs)
    # 2.punctation tfidf
    punct_vectorizer = TfidfVectorizer(tokenizer=punct_tokenize)
    punct_vectors = punct_vectorizer.fit_transform(paragraphs)
    # 3.POS tfidf
    pos_vectorizer = TfidfVectorizer(tokenizer=pos_tagger)
    pos_vectors = pos_vectorizer.fit_transform(paragraphs)
    # 4.stopwords tfidf
    stopword_vectorizer = TfidfVectorizer(tokenizer=stopword_tokenize)
    stopword_vectors = stopword_vectorizer.fit_transform(paragraphs)
    # 5.3-grams tfidf
    three_gram_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(3, 3))
    three_gram_vectors = three_gram_vectorizer.fit_transform(paragraphs)
    return [word_vectors, punct_vectors, pos_vectors, stopword_vectors, three_gram_vectors]


def make_prediction(feature_vectors, paragraphs, hyperparams):
    # style fingerprints by features
    if not feature_vectors:
        classic_features_vecs = get_classic_feature_vecs(paragraphs)
        vectors = sp.hstack(classic_features_vecs, format='csr')
        denselist = (vectors.todense()).tolist()
        df = pd.DataFrame(data=denselist)
    else:
        df = pd.DataFrame(data=feature_vectors)

    # borders by Wilcoxon test
    style_change_borders = find_style_change_starts_by_wilcoxon(features_data_frame=df,
                                                                hyperparams=hyperparams)

    # result format correction(found borders -> breaches)
    style_breaches = [] if not style_change_borders else get_start_indices(style_change_borders, paragraphs)
    result_dict = {"style_change": True if style_change_borders else False, "style_breaches": style_breaches}
    return result_dict
