import jsonpickle
import json
import pickle
import numpy as np
import os

from .features import lexical
from .features.word_frequency import WordFrequency
from .transformers import phrase_frequency
from .transformers import frequent_words_diff
from .features import readability
from .transformers import max_diff
from .transformers import text_length


def load_model(path_to_model):
    predictors = []
    scalers = []
    with open(os.path.join(path_to_model, 'predictors.pckl'), 'rb') as f:
        while True:
            try:
                predictors.append(pickle.load(f))
            except:
                break
    with open(os.path.join(path_to_model, 'scalers.pckl'), 'rb') as f:
        while True:
            try:
                scalers.append(pickle.load(f))
            except:
                break
    with open(os.path.join(path_to_model, 'model.json'), 'r') as f:
        js = json.load(f)
        model = jsonpickle.decode(js)

    phrase_frequency_func = lambda x: np.array(phrase_frequency(x, **model.params['phrase_transformer']))
    frequent_words_diff_func = lambda x: np.array(frequent_words_diff(x, **model.params['frequent_words_diff_transformer']))
    min_max_lexical_per_segment_func = lambda x: max_diff(lexical(x))
    rare_richness_func = lambda x: max_diff(WordFrequency().average_word_frequency((x)))
    text_length_func = lambda x: np.array(text_length(x))
    readability_func = lambda x: np.array(max_diff(readability(x)))

    model.stack = [
        (min_max_lexical_per_segment_func, True, True, scalers[0], predictors[0]),
        (rare_richness_func, True, False, scalers[1], predictors[1]),
        (phrase_frequency_func, False, False, scalers[2], predictors[2]),
        (frequent_words_diff_func, False, False, scalers[3], predictors[3]),
        (readability_func, True, False, scalers[4], predictors[4]),
        (text_length_func, False, False, scalers[5], predictors[5])
    ]

    return model