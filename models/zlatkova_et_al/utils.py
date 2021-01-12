import os
import json
from itertools import zip_longest
from time import gmtime, strftime
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import spacy_udpipe
from collections import Counter
import math
import re
import stanza
from typing import List, Tuple


nlp = spacy_udpipe.load(lang='hy')
nlp_stanza = stanza.Pipeline(lang='hy', processors='tokenize, pos, lemma')


def lemmatizer(text):
    doc = nlp_stanza(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]


def pos_tagger(text):
    doc = nlp_stanza(text)
    return [word.pos for sentence in doc.sentences for word in sentence.words]


def word_tokenize(text, remove_punctuation=False):
    text = remove_punct(text) if remove_punctuation else text
    doc = nlp(text)
    return [word.text for word in doc]


def sentence_tokenizer(text):
    doc = nlp(text)
    return [x.string for x in list(doc.sents)]


def letter_tokenize(text):
    return list(re.sub(r'[^u0561-\u0587\u0531-\u0556]', '', text))


def letters_and_numbers(text):
    return list(re.sub(r'[^\d\u0561-\u0587\u0531-\u0556]', '', text))


def remove_punct(text):
    return re.sub(r'[^\d\s\u0561-\u0587\u0531-\u0556]', ' ', text)


def syllables_counter(text):
    c = Counter(text)
    return c["ա"] + c["ե"] + c["է"] + c["ը"] + c["ի"] + c["ո"] + c["և"] + c["օ"]


def readability_arm(text):
    words = word_tokenize(text, remove_punctuation=True)
    words_count = len(words)
    syllables_count = syllables_counter(text)
    sentences_count = len(sentence_tokenizer(text))
    char_count = len(letter_tokenize(text))
    return [flesch_reading_ease(words_count, syllables_count, sentences_count),
            smog_index(syllables_count, sentences_count),
            flesch_kincaid_grade(words_count, syllables_count, sentences_count),
            coleman_liau_index(char_count, words_count, sentences_count),
            automated_readability_index(len(letters_and_numbers(text)), words_count, sentences_count),
            dale_chall_readability_score(words, sentences_count),
            linsear_write_formula(words), difficult_words(words), gunning_fog(words, sentences_count)]


def flesch_reading_ease(words_count, syllables_count, sentences_count):
    try:
        FSE = 78.39 + 2.6 * (words_count / sentences_count) - 32.3 * (syllables_count / words_count)
    except ZeroDivisionError:
        return 0.0
    return round(FSE, 2)


def smog_index(syllables_count, sentences_count):
    try:
        smog = 0.6 * math.sqrt(syllables_count / sentences_count) + 9.0
    except ZeroDivisionError:
        return 0.0
    return round(smog, 2)


def flesch_kincaid_grade(words_count, sentences_count, syllables_count):
    try:
        FK = -0.33 * (words_count / sentences_count) + 6.42 * (syllables_count / words_count) + 4.7
    except ZeroDivisionError:
        return 0.0
    return round(FK, 2)


def coleman_liau_index(letters_count, words_count, sentences_count):
    try:
        CL = 1.2 * (letters_count / words_count) + 62.65 * (sentences_count / words_count) + 0.662
    except ZeroDivisionError:
        return 0.0
    return round(CL, 2)


def automated_readability_index(letters_and_nums_count, words_count, sentences_count):
    try:
        AT = 3.062 * (letters_and_nums_count / words_count) - 0.049 * (words_count/sentences_count) + 0.078
    except ZeroDivisionError:
        return 0.0
    return round(AT, 2)


def dale_chall_readability_score(words, sentences_count):
    words_count = len(words)
    c = 0
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    count = words_count - c
    try:
        per = float(count) / float(words_count) * 100
    except ZeroDivisionError:
        return 0.0
    difficult_words = 100 - per
    score = ((0.1579 * difficult_words) + (0.0496 * words_count / sentences_count))
    if difficult_words > 5:
        score += 3.6365
    return round(score, 2)


def linsear_write_formula(words):
    words = words[:100]
    sentences_count = len(sentence_tokenizer(' '.join(words)))
    c1 = 0
    c3 = 0
    for word in words:
        if syllables_counter(word) < 3:
            c1 = c1 + 1
        else:
            c3 = c3 + 1
    try:
        lin = float((c1 + c3) / sentences_count)
    except ZeroDivisionError:
        return 0.0
    return round(lin, 2)


def difficult_words(words):
    c = 0.0
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    return c


def gunning_fog(words, sentences_count):
    words_count = len(words)
    c = 0
    for word in words:
        if syllables_counter(word) > 3:
            c += 1
    try:
        GF = 0.4 * (words_count / sentences_count + 100 * (c / words_count))
    except ZeroDivisionError:
        return 0.0
    return round(GF, 2)


def print_splits(texts, positions):
    text_colors = ['1;31', '1;32', '1;33', '1;34']

    whole_print = []

    for index, text in enumerate(texts):
        positions[index].append(len(text))
        text_marker = 0
        local_print = ''

        for color_index, change in enumerate(positions[index]):
            local_print += '\x1b[%sm%s\x1b[0m' % (text_colors[color_index], text[text_marker:change])
            text_marker = change

        whole_print.append(local_print)

    print('\n\n=============================================\n\n'.join(whole_print))


def split_data(documents: List[Tuple[str, dict]]):
    x, y, positions = [], [], []
    for data in documents:
        x.append(data[0])
        y.append(data[1]['style_change'])
        positions.append(data[1]['style_breaches'])
    return x, y, positions


def get_data(main_dir=None, external_file=None, breach=False):
    x, y, positions, file_names = [], [], [], []
    if main_dir:
        x, y, positions, file_names = get_data_from_dir(main_dir, breach)

    if external_file:
        data = pd.read_feather(external_file)
        external_x = data['text'].values.tolist()
        external_y = [len(x) > 0 for x in data['positions']]
        external_positions = [map(int, x.split(',')) for x in data['positions']]

        x += external_x
        y += external_y
        positions += external_positions

    return x, y, positions, file_names


def get_external_data(file, train_size, val_size):
    data = pd.read_feather(file)
    X = data['text'].values.tolist()
    y = [len(x) > 0 for x in data['positions']]
    
    return train_test_split(X, y, stratify=y, train_size=train_size, test_size=val_size, random_state=2)


def get_data_from_dir(directory, breach=False, size=None):
    x = []
    y = []
    positions = []
    file_names = []
    n = 0

    for entry in os.listdir(directory):
        if n == size:
            break

        root, ext = os.path.splitext(entry)
        if ext == '.txt':
            with open(os.path.join(directory, ''.join([root, ext])), encoding='utf8') as txt_file:
                text = txt_file.read()
                x.append(text)
                file_names.append(root)
                n += 1
            try:
                with open(os.path.join(directory, ''.join([root, '.truth'])), encoding='utf8') as truth_file:
                    truth = json.load(truth_file)
                    if breach:
                        truth_changes = len(truth['borders']) > 0
                        truth_positions = truth['borders']
                    else:
                        truth_changes = truth['changes']
                        truth_positions = truth['positions']
                    y.append(truth_changes)
                    positions.append(truth_positions)
            except IOError:
                pass

                if(size):
                    print_progress_bar(n, size, description = 'Loading artificial data')

    return x, y, positions, file_names


def get_results(train_size, clf_params, cv=None, val=None, gs=None):
    if cv:
        cv = {
            'train_score': {
                "mean": round(np.mean(cv['train_score']), 4),
                "std": round(np.std(cv['train_score']), 2)
            },
            'test_score': {
                "mean": round(np.mean(cv['test_score']), 4),
                "std": round(np.std(cv['test_score']), 2),
                "all": round_np_scores(cv['test_score'], 4)
            },
            'fit_time': humanize_time(max(cv['fit_time'])),
            'score_time': humanize_time(max(cv['score_time']))
        }

    if val:
        val = {
            'accuracy': round(val['accuracy'], 4),
            'time': humanize_time(val['time'])
        }

    results = {
        'cross_validation': cv,
        'validation': val,
        'grid_search': gs,
        'estimator': clf_params,
        'train_size': train_size,
        'timestamp': strftime("%Y-%m-%d %H:%M:%S", gmtime())
    }

    json_results = json.dumps(results, indent=4, sort_keys=True)
    return json_results


def write_results_to_file(results):
    output_separator = '================================================='

    results_path = config_local().get('results_file', None)

    if not results_path:
        print('No file name specified for results!')
        return

    with open(results_path, 'a') as output:
        output.write('%s\n%s\n' % (results, output_separator))

def config_local():
    with open('config.json', 'r') as config_json:
        return json.load(config_json)

def persist_output(output_dir, predictions, file_names, breach=False):
    for prediction, file_name in zip(predictions, file_names):
        if breach:
            prediction = {
                'borders': prediction
            }
        else:
            tag = True if prediction == 1 else False

            prediction = {
                'changes': tag
            }

        json_prediction = json.dumps(prediction, indent=8)

        with open('%s/%s.truth' % (output_dir, file_name), 'w') as output:
            output.write(json_prediction)


def round_np_scores(np_array, p=None):
    return [round(x, p) for x in np_array.tolist()]


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)

    return '%02d:%02d:%02d' % (hours, mins, secs)


def print_progress_bar(iteration, total, description='', decimals=1, bar_length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + '-' * (bar_length - filled_length)

    print('\r |%s| %s%% | %s' %
          (bar, percent, description), end='\r')

    if iteration == total:
        print()


def chunker(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n

    return list(zip_longest(*args, fillvalue=fillvalue))


def get_n_jobs():
    with open('config.json', 'r') as rc:
        n_jobs = json.load(rc).get('n_jobs', 1)

        return n_jobs

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-i", dest="input_dir", help="input_dir", metavar="FILE")
    parser.add_argument("-o", dest="output_dir", help="output_dir", metavar="FILE")
    args = parser.parse_args()

    return args.input_dir, args.output_dir

def update_dict(params, keys, value):
        if len(keys) == 1:
            params[keys[0]] = value
        else:
            update_dict(params[keys[0]], keys[1:], value)


get_readability_test = {
    'flesch_reading_ease': flesch_reading_ease,
    'smog_index': smog_index,
    'flesch_kincaid_grade': flesch_kincaid_grade,
    'coleman_liau_index': coleman_liau_index,
    'automated_readability_index': automated_readability_index,
    'dale_chall_readability_score': dale_chall_readability_score,
    'difficult_words': difficult_words,
    'linsear_write_formula':linsear_write_formula,
    'gunning_fog': gunning_fog
}