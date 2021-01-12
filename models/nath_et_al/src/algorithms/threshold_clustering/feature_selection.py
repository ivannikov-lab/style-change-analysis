import preprocess_NLP_pkg
from math import log
import re
from .cluster_graph import *


def calculate_words_wfm(windows, selected_features):
    """Splits a text into windows of given size/ step size, converts windows into feature matrix against a given set of selected word features
        Keyword arguments:
            text -- the text to be converted into window feature matrix
            selected_features -- the features against for which the feature vector must be generated
    """
    window_feature_matrix = []
    for window in windows:
        word_freq = preprocess_NLP_pkg.word_freq_count_normalised(window)
        selected_word_freq = preprocess_NLP_pkg.select_feature_vector(word_freq, selected_features)
        selected_word_vector = list(selected_word_freq.values())
        window_feature_matrix.append(selected_word_vector)
    return np.asarray(window_feature_matrix)


def calculate_ngrams_wfm(windows, selected_features, n):
    """Splits a text into windows of given size/ step size, converts windows into feature matrix against a given set of selected character ngrams
        Keyword arguments:
            text -- the text to be converted into window feature matrix
            selected_features -- the features against for which the feature vector must be generated
            n -- n in ngrams
    """
    window_feature_matrix = []
    for window in windows:
        ngram_freq = preprocess_NLP_pkg.char_ngram_count_normalised(window, n=n)
        selected_ngram_freq = preprocess_NLP_pkg.select_feature_vector(ngram_freq, selected_features)
        selected_ngram_vector = list (selected_ngram_freq.values())
        window_feature_matrix.append(selected_ngram_vector)
    return np.asarray(window_feature_matrix)


def calculate_window_distance(wfm, distance_measure):
    """Given a window feature matrix and a distance measure, returns a distance matrix by calculating distances between the window feature vectors
            Keyword arguments:
                wfm -- window feature matrix
                distance_measure -- any distance function (ideally matusita/ tanimoto)
    """
    dist_m = np.zeros((wfm.__len__(),wfm.__len__()))
    for i in range(0,wfm.__len__()):
        window1 = wfm[i]
        for j in range(0,wfm.__len__()):
            window2 = wfm[j]
            if i!=j:
                try:
                    dist_m[i][j] = distance_measure(window1, window2)
                except NameError:
                    print("The function ",distance_measure, " does not exist! Returning None")
                    return None
    return dist_m


def remove_empty_para(para_list):
    """Given a list of paragraphs, returns a list of paragraphs that are not empty or have chars greater than 10 characters
        Keyword arguments:
            text_list -- a list of texts
    """
    return [para for para in para_list if not(re.match("| |\n| \n|\n\n|\n \n",para) and para.__len__()<100) ]


def para_size_greater_than_n(para_list, n = 0):
    """Given a list of texts, prints the number of characters of each paragraph.
        Keyword arguments:
            text_list -- a list of texts
            n -- return paragraphs of size > n characters
    """
    #for para in para_list:
    #    print(len(para))
    if n > 0:
        return [para for para in para_list if len(para)>n]


def rank_matrix(m, rank_matrix = None):
    """ Given a matrix, calculates the rank of each element and updates a rank matrix (or creates a new one if not present)
    The rank matrix is required when aggregating the ranks of different distance matrices.
        Keyword arguments:
            m -- a distance matrix
            rank_matrix -- a rank matrix
    """
    if rank_matrix is None:
        rank_matrix = np.zeros((m.__len__(),m.__len__() ))
    elem_list = list(set(m.flatten()))
    elem_list.sort()
    #print(elem_list)
    for i in range(0, m.__len__()):
        for j in range(0, m.__len__()):
            if i < j:
                rank = elem_list.index(m[i][j])
                #print(rank, rank_matrix[i][j])
                rank_matrix[i][j] = rank_matrix[i][j] + rank
    return rank_matrix


def get_combined_rank(matusita_mfw, matusita_mfng, tanimoto_mfw, tanimoto_mfng):
    """ Given different distance matrices, combines the rank of all of them into a single rank matrix.
        Keyword arguments:
            matusita_mfw -- matusita distance matrix with words
            matusita_mfng -- matusita distance matrix with N grams
            tanimoto_mfw -- tanimoto distance matrix with words
            tanimoto_mfng -- tanimoto distance matrix with N grams
    """
    rank_m = rank_matrix(matusita_mfw)
    rank_m = rank_matrix(matusita_mfng, rank_m)
    rank_m = rank_matrix(tanimoto_mfw, rank_m)
    rank_m = rank_matrix(tanimoto_mfng, rank_m)
    return rank_m


def get_order_log_dist(n):
    """ Calculate the log distance matrix between windows from their positions at a distance matrix.
        For example, distance between w0 and w10 is log(9)
        Keyword arguments:
            n: number of windows
    """
    order_dist_m = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            if(j>i):
                order_dist_m[i][j] = log(j-i)
    return order_dist_m
