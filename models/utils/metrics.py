# Natural Language Toolkit: WinPR
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Martin Scaiano <martin@scaiano.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
##########################################################################
# WinPR
# Scaiano, M., Inkpen, D.
# Getting More from Segmentation Evaluation
# NAACL HLT 2012, pp. 362-366
##########################################################################

import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def windowsize(gold, boundary="1"):
    return len(gold) / gold.count(boundary)


class Metrics:
    tp = 0.0  # True Positives
    tn = 0.0  # True Negatives
    fp = 0.0  # False Positives
    fn = 0.0  # False Negatives

    k = 1  # window size

    def __init__(self, gold_seg, hypo_seg, k, boundary="1"):
        self.k = k
        if len(gold_seg) != len(hypo_seg):
            sys.exit("Segmentations have unequal length")
        if self.k != 0:
            for i in range(len(gold_seg) + 1 - self.k):
                self.update(gold_seg[i:i + k].count(boundary), hypo_seg[i:i + k].count(boundary))
        else:
            self.update(gold_seg[0].count(boundary), hypo_seg[0].count(boundary))

    def precision(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def accuracy(self):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fn + self.fp)

    def f1score(self, beta=1.0):
        if (self.precision() == 0 and self.recall() == 0) or self.precision() < 0 or self.recall() < 0:
            return 0.0
        return (1.0 + beta) * (self.precision() * self.recall() / (beta ** 2 * self.precision() + self.recall()))

    def update(self, goldCount, hypoCount):
        self.tp += min(goldCount, hypoCount)
        self.tn += self.k - max(goldCount, hypoCount)if self.k > 0 else 0.0
        self.fn += max(0, goldCount - hypoCount)
        self.fp += max(0, hypoCount - goldCount)

    def write(self):
        print("tp: ", self.tp)
        print("tn: ", self.tn)
        print("fp: ", self.fp)
        print("fn: ", self.fn)
        print("precision: ", self.precision())
        print("recall: ", self.recall())


def metrics(truth, presictions):
    winP = precision_score(truth, presictions)
    winR = recall_score(truth, presictions)
    winF = f1_score(truth, presictions)
    acc = accuracy_score(truth, presictions)

    return winP, winR, winF, acc