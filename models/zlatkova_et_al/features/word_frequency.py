import re
import math
from os.path import dirname, abspath, join
from ..utils import print_progress_bar, lemmatizer, word_tokenize

dir = dirname(dirname(abspath(__file__)))
COMMON_WORDS_FILE = join(dir, 'data/external/common_words/most_common_arm.txt')


class WordFrequency():
    def __init__(self):
        self.word_class = {}
        with open(COMMON_WORDS_FILE) as f:
            for line in f:
                key, val = line.split()
                self.word_class[key.lower()] = math.log2(3438126/float(val))

    def average_word_frequency(self, X, feature_names=[]):
        transformed = []

        for i, data in enumerate(X):
            words = word_tokenize(data[0])
            segs = data[1]
            segments = []

            for start, end in segs:
                entry = ' '.join(words[start:end])
                class_sum = 0
                word_count = 0
                uncommon = 0
                entry = lemmatizer(entry)
                for w in entry:
                    w = w.lower()
                    w = re.sub('[^\u0561-\u0587\u0531-\u0556]+', '', w)
                    if not w: continue

                    word_count+=1
                    word_class = self.word_class.get(w, 20)
                    if word_class == 20:
                        uncommon += 1
                    class_sum += word_class

                try:
                    segments.append([class_sum/word_count, uncommon/word_count])
                except ZeroDivisionError:
                    segments.append([-1, -1])


            transformed.append(segments)
            print_progress_bar(i + 1, len(X), description = 'word frequency')

        feature_names.extend(['average_word_class', 'uncommon_words'])

        return transformed
