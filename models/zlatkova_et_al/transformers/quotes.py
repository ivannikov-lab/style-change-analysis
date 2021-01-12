from ..utils import print_progress_bar

def quote_discrepancies(data, feature_names=[]):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        entry = entry.lower()

        count_single = entry.count("\'")
        count_double = entry.count("\"")
        count_quotes_open = entry.count("«")
        count_quotes_close = entry.count("»")

        vectors.append([float(min(count_single, count_double, count_quotes_open, count_quotes_close))])

        print_progress_bar(i + 1, data_length, description = 'quote_discrepancies')

    feature_names.extend(['quote_discrepancies'])

    return vectors
