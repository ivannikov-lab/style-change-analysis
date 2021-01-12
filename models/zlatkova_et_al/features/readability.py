from ..utils import print_progress_bar, readability_arm, word_tokenize

# def readability(X, indices = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index', \
#      'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', \
#      'linsear_write_formula', 'gunning_fog'], feature_names = []):
def readability(X, indices = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index', \
     'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', \
     'linsear_write_formula', 'gunning_fog'], feature_names = []):
    transformed = []

    data_length = len(X)

    for i, data in enumerate(X):
        words = word_tokenize(data[0])
        segs = data[1]
        segments = []

        for start, end in segs:
            entry = ' '.join(words[start:end])
            segments.append(readability_arm(entry))

        transformed.append(segments)

        print_progress_bar(i + 1, data_length, description='readability')

    feature_names.extend(indices)

    return transformed
