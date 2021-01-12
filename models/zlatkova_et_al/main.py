import time
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from nltk import ConfusionMatrix
import numpy as np

from .utils import get_data, get_n_jobs, get_arguments, split_data
from .models import StackingSimple
from .chunkers import get_sentences

TRAINING_DIR = 'data/training'
VALIDATION_DIR = 'data/validation'

input_dir, output_dir = get_arguments()
# TEST_DIR = input_dir or 'data/test'
OUTPUT_DIR = output_dir or 'data/output'


def train(train_data, valid_data,
         estimator=StackingSimple,
         cv_split=10,
         with_cross_validation=False,
         with_validation=False,
         with_full_data_tfidf=False,
         train_with_validation=True,
         path_to_save_model=""):


    train_x, train_y, train_positions = split_data(train_data)

    if train_with_validation:
        validation_x, validation_y, validation_positions = split_data(valid_data)
        train_x.extend(validation_x)
        train_y.extend(validation_y)

    clf, cv, val, gs = None, None, None, None

    if estimator:
        clf = estimator()

    if with_cross_validation:
        if with_full_data_tfidf:
            skf = StratifiedKFold(n_splits=cv_split, random_state=42, shuffle=True)
            all_acc = []
            X = np.array(train_x)
            y = np.array(train_y)
            for train_index, test_index in skf.split(X, y):
                y_train, y_test = y[train_index], y[test_index]
                X_train, X_test = X[train_index], X[test_index]
                print(X_train.shape)

                clf.fit_with_test(X_train.tolist(), y_train, train_positions, X_test.tolist())
                predictions = clf.predict(X_test.tolist())
                all_acc.append(accuracy_score(y_test, predictions))

            print("Accuracies:", all_acc)
            print("Mean:", np.mean(all_acc))
            print("Stdev:", np.std(all_acc))

        else:
            cv = cross_validate(estimator=clf, X=train_x, y=train_y, fit_params={'train_positions': train_positions}, cv=cv_split,
                            scoring="accuracy", n_jobs=get_n_jobs(), return_train_score=True)


    if with_validation:
        t_start = time.time()
        if with_full_data_tfidf:
            clf.fit_with_test(train_x, train_y, train_positions, validation_x)
        else:
            clf.fit(train_x, train_y, train_positions)

        predictions = clf.predict(validation_x)
        t_end = time.time()

        print(ConfusionMatrix(validation_y, predictions))

        val = {
            'accuracy': accuracy_score(validation_y, predictions),
            'time': t_end - t_start
        }
        print(val['accuracy'])
    else:
        clf.fit(train_x, train_y)
    clf.save_model(path_to_save_model)


def get_breach_predictions(clf, test_x, change_predictions):
    predictions = []
    for has_change, text in zip(change_predictions, test_x):
        if has_change:
            sentences = get_sentences(text)
            breaches = find_breaches(clf, sentences, 0, len(sentences))
            print('BREACHES: ', breaches)
            predictions.append(breaches)
        else:
            predictions.append([])

    return predictions


def find_breaches(clf, sentences, l, r):
    x = np.expand_dims(' '.join(sentences[l:r]), axis=0)
    has_change = clf.predict(x)[0]

    if not has_change:
        return []

    if r - l <= 10:
        return [len(' '.join(sentences[:(l+r)//2]))]
    else:
        mid = (r-l) // 2
        left = find_breaches(clf, sentences, l, l+mid)
        right = find_breaches(clf, sentences, l+mid, r)
        if len(left) == 0 and len(right) == 0:
            return [len(' '.join(sentences[:l+mid]))]

        return left + right


def test(clf, train_x, train_y, train_positions, with_full_data_tfidf, test_x):
    if len(test_x) <= 0: return print('Test dataset is empty!')
    t_start = time.time()
    if with_full_data_tfidf:
        clf.fit_with_test(train_x, train_y, train_positions, test_x)
    else:
        clf.fit(train_x, train_y, train_positions)
    predictions = clf.predict(test_x)
    t_end = time.time()
    return predictions


def grid_search(clf, params, x, y, positions):
    gs_clf = GridSearchCV(clf, params, n_jobs=get_n_jobs(), scoring='accuracy', verbose=1, cv=2)
    gs_clf.fit(x, y, positions)

    print("Best parameters:")
    print(gs_clf.best_params_)
    print("Best score: %0.3f" % gs_clf.best_score_)

    return gs_clf.best_estimator_, gs_clf.best_score_
