"""This file executes the threshold based clustering method over a training text
"""

import preprocess_NLP_pkg
from .feature_selection import calculate_words_wfm, \
    calculate_window_distance,calculate_ngrams_wfm
from .cluster_graph import get_cg_avg_dist, prune_clusters, cluster_graph
from ..preprocessing import paragraph_tokenizer
import numpy as np
from scipy import stats
import nltk
from ..preprocessing import  remove_duplicate_sentences, is_duplicated


def token_count_analyse(text, selected_words):
    token_count = 0
    tokens = nltk.tokenize.word_tokenize(text.lower())
    for word in selected_words:
        temp = tokens.count(word)
        token_count = token_count + temp
        #print(word, temp)
        #token_count = len(re.findall(token, text.lower()))
        #print(file, token_count/len(tokens))
    return  token_count/len(tokens)


def extra_param(text):
    ttr = preprocess_NLP_pkg.ttr(text)
    i_list = ['i', 'my', 'mine', 'me', 'myself']
    you_list = ['you', 'your', 'yours', 'yourself', 'yourselves']
    you_freq = token_count_analyse(text, you_list)
    i_freq = token_count_analyse(text, i_list)
    if ttr < 15:
        return 2
    elif you_freq > 0.04 or ttr> 45 : #or i_freq < 0.06
        return 1
    else:
        return -1


def execute_threshold_clustering(text, merge_threshold, add_node_threshold, prune = True, number_of_terms=50, distance_measure = preprocess_NLP_pkg.matusita_distance, use_duplication_feature = True):
    """
    From a text, selects the word features, creates windows, calls the clustering method to create a list of
    cluster graphs.
    :param text: text for which clustering is to be determined
    :param merge_threshold: Two clusters can be merged together if the average distance of the new cluster is below
             the merge_thresholds for either contributing clusters
    :param add_node_threshold: A new node can be added to a cluster if the average distance of the new cluster is below
             the add_node_threshold of the cluster
    :param prune: boolean for pruning the cluster list, prune = True causes the clusters to be smaller???
    :param number_of_terms: the number of selected features to be used
    :param distance_measure: the type of distance measure to be used
    :param use_duplication_feature: Boolean for whether to use duplication for prediction
    :return: number of authors
    """
    if use_duplication_feature:
        if is_duplicated(text) < 2:
            number_of_authors = 1
            return number_of_authors
    #text = remove_duplicate_sentences(text)
    #number_of_authors = extra_param(text)
    #if number_of_authors > 0:
    #    return number_of_authors
    selected_words_freq_dist = preprocess_NLP_pkg.word_freq_count(text, number_of_terms=number_of_terms)
    selected_words = list(selected_words_freq_dist.keys())
    #selected_ngrams_freq_dist = preprocess_NLP_pkg.char_ngram_count(text, n=4, number_of_terms=50)
    #selected_ngrams = list(selected_ngrams_freq_dist.keys())
    windows = paragraph_tokenizer(text, remove_url=True, remove_empty_paragraphs=True, balance_large_paragraphs=True)
    # windows = preprocess_NLP_pkg.window_tokenizer(text, window_size = 2000, step_size = 2000)
    if len(windows) <= 1:
        number_of_authors = 1
        #print(100, ",\t", 100)
    else:
        words_fm = calculate_words_wfm(windows, selected_words)
        #char_ngrams_fm = calculate_ngrams_wfm(windows, selected_ngrams, n=4)
        dist_matrix = calculate_window_distance(words_fm, distance_measure= distance_measure)
        #threshold = find_entries_outside_threshold(result, n=1)
        #if threshold < 0.1:
        #    number_of_authors = 1
        #    return number_of_authors
        #    print(100, ",\t", 100)
        #else:
        #flat = np.array([val for val in result.flatten() if val > 0])
        #min = flat.min()
        # max,min,mean,median,truncated mean,std,threshold,file,observed,pred
        #print(result.max(),",",min,",",result.mean(),",",np.median(result),",",preprocess_NLP_pkg.stats.trunc_mean(result), ",",result.std(),",", find_entries_outside_threshold(result, n=1)) # np.where(result == result.max())[0]
        node_labels = range(0, len(dist_matrix))
        cg_list = cluster_graph(node_labels, dist_matrix, merge_cluster_threshold=merge_threshold,
                                add_node_threshold=add_node_threshold)
        if prune:
            prune_clusters(cg_list)
        number_of_authors = reduce_excess_authors(cg_list) # reduces authors when there are too many cluster graphs, say more than 5
    return number_of_authors


def find_entries_outside_threshold(mat, n = 1.0):
    return (mat > (mat.mean() + n * mat.std())).sum()/mat.size

def reduce_excess_authors(cg_list):
    """
    If the size of the cg_list > 5 and if all the clusters have avg dist > 0.7, then designate number of authors to be 1
    :param cg_list:
    :return:
    """

    cg_dist_greater_than_threshold = sum([1 for cg in cg_list if get_cg_avg_dist(cg) > 0.7])
    if cg_dist_greater_than_threshold == len(cg_list) and len(cg_list) > 5:
        return 1
    else:
        return len(cg_list)


'''
def get_observed_authors(folder_path):
    """ Method Description
            Keyword arguments:
                arg1: description
        """
    observed_authors = []
    ground_truth_files = preprocess_NLP_pkg.load_files_from_dir(folder_path, '*.truth')
    for file in ground_truth_files:
        with open(folder_path + "/" + file) as f:
            data = json.load(f)
            authors = int(data["authors"])
            observed_authors.append(authors)
    return np.asarray(observed_authors)
'''


'''
#run_cluster_graph(text_path, merge_thresholds = [5,10, 50,100,500,1000], add_node_thresholds = [5,10, 50,100,500,1000])
folder_path = '/Users/sukanyanath/Documents/PhD/CLEF/pan19-style-change-detection-training-dataset-2019-01-17/'

observed_authors = get_observed_authors(folder_path)
predicted_authors = run_cluster_graph(folder_path, observed_authors, merge_thresholds = 50, add_node_thresholds = 100, prune= True)
cm = confusion_matrix(observed_authors, predicted_authors)
print(cm)
acc = get_accuracy(cm)
print("Accuracy: ", acc)
oci = get_oci(cm, K= cm.shape[0])
print("OCI: ", oci)
print("Rank: ", (acc+ 1-oci)/2)
# index_1 = [i for i,k in enumerate(observed_authors) if observed_authors[i]!=predicted_authors[i] and observed_authors[i]==1]
'''

'''
def run_cluster_graph(folder_path, observed_authors, merge_thresholds, add_node_thresholds, prune):
    files = preprocess_NLP_pkg.load_files_from_dir(folder_path, '*.txt')
    predicted_authors = []
    for i in range(0, len(files)):
        file = files[i]
        #print("\n\n\n Evaluating ", file)
        training_text = preprocess_NLP_pkg.read_file(folder_path + "/" + file, mode='rb', ignore_comments=False).decode('utf-8')
        training_text = remove_duplicate_sentences(training_text)
        #print("\n\nFeature selection started ")
        selected_words_freq_dist = preprocess_NLP_pkg.word_freq_count(training_text, number_of_terms=50)
        selected_words = list(selected_words_freq_dist.keys())
        selected_ngrams_freq_dist = preprocess_NLP_pkg.char_ngram_count(training_text, n= 4, number_of_terms=50)
        selected_ngrams = list(selected_ngrams_freq_dist.keys())
        windows = paragraph_tokenizer(training_text, remove_URL = True, remove_empty_paragraphs = True)
        #windows = group_texts(windows,2)
        #print("Number of windows: ", windows.__len__()) # calculate the window feature matrices
        if len(windows) <= 1:
            #print("Since only one window, therefore 1 author")
            predicted_authors.append(1)
            print(file, observed_authors[i], predicted_authors[i])
        else:
            #print("Length of each window")
            #for i in range(0, len(windows)):
            #    print("W",i,": ",len(windows[i]))
            #print("Feature selection completed ")
            #print("\n\nWindow Feature Matrix creation started ")
            words_fm = calculate_words_wfm(windows, selected_words)
            char_ngrams_fm = calculate_ngrams_wfm(windows, selected_ngrams, n=4)
            #print("Window Feature Matrix creation completed ")
            #print("\n\nDistance matrix creation started")
            result_matusita_words = calculate_window_distance(words_fm, preprocess_NLP_pkg.matusita_distance)
            #result_matusita_ngrams = calculate_window_distance(char_ngrams_fm,preprocess_NLP_pkg.matusita_distance)
            #result_tanimoto_words = calculate_window_distance(words_fm, preprocess_NLP_pkg.tanimoto_distance)
            #result_tanimoto_ngrams = calculate_window_distance(char_ngrams_fm, preprocess_NLP_pkg.tanimoto_distance)
            #rank_combined = get_combined_rank(result_matusita_words, result_matusita_ngrams, result_tanimoto_words,result_tanimoto_ngrams)
            #print("Distance matrix creation completed")
            node_labels = range(0,len(result_matusita_words))
            #print("\n\nCluster graph creation started")
            cg_list = cluster_graph(node_labels, result_matusita_words, merge_cluster_threshold = merge_thresholds, add_node_threshold = add_node_thresholds)
            #print("Number of Clusters:", len(cg_list))
            if prune:
                prune_clusters(cg_list)
            number_of_authors = reduce_excess_authors(cg_list) # len(cg_list) #
            predicted_authors.append(number_of_authors)
            print(file, observed_authors[i], predicted_authors[i])
            #cg_list_list = iterate(node_labels, result_matusita_words, merge_thresholds, add_node_thresholds, prune)
            #print("Cluster graph creation ended")
    return np.array(predicted_authors)

'''