import re
import nltk
from statistics import mean, stdev
from math import ceil
import collections
from ...utils import sentence_tokenizer


def paragraph_tokenizer(text="", remove_url=True, remove_empty_paragraphs=True, balance_large_paragraphs=True):
    """
    Tokenizes text into paragraph sized chunks, may also be used to create modified windows by setting group_texts
    to True
    :param text: the text to tokenize
    :param remove_url: boolean to remove URLs
    :param remove_empty_paragraphs: boolean to remove empty paragraphs
    :param balance_large_paragraphs: boolean to split large paragraphs into smaller ones
    :return:
    """
    if remove_url:
        text = re.sub("[\s]+\$URL\$", "", text) # remove URLs
    paragraphs = re.split('։ \n|։\n', text)
    paragraphs = list(set(paragraphs))
    if len(paragraphs) == 1 :
        paragraphs = sentence_tokenizer(text)
        paragraphs = group_texts(paragraphs,10)
        return paragraphs
    if balance_large_paragraphs:
        paragraphs = split_large_paragraphs(paragraphs)
    if remove_empty_paragraphs:
        paragraphs =remove_empty_para(paragraphs)
    return paragraphs


def split_large_paragraphs(paragraphs):
    """
    Identifies large paragraphs (len(para) > mean(avg para length + 3*sd (avg para length) and breaks them into
    smaller paragraphs of 5 sentences each.
    :param paragraphs:
    :return:
    """
    para_length = [len(para) for para in paragraphs]
    mean_para_length = mean(para_length)
    sd_para_length = stdev(para_length)
    new_paragraphs = []
    if len(paragraphs)>5:
        threshold = mean_para_length + sd_para_length
    else:
        threshold = mean_para_length
    for i in range(0, len(paragraphs)):
        para = paragraphs[i]
        if len(para) > threshold:
            splits = sentence_tokenizer(para)
            splits = group_texts(splits, 5)
            new_paragraphs = new_paragraphs + splits
        else:
            new_paragraphs.append(para)
    return new_paragraphs


def remove_empty_para(para_list):
    """
    Given a list of paragraphs, returns a list of paragraphs that are not empty or have chars greater than 200
    characters
    :param para_list: a list of paragraphs
    :return: a list of non-empty paragraphs having length at least 200 chars
    """
    return [para for para in para_list if not(re.match("| |\n| \n|\n\n|\n \n|։ \n",para) and para.__len__()<200) ]


def para_size_greater_than_n(para_list, n = 1):
    """
    Returns paragraphs whose length are greater than n
    :param para_list: a list of paragraphs
    :param n: paragraphs having length >n are selected
    :return: list of paragraphs having length >n
    """
    if n > 0:
        return [para for para in para_list if len(para)>n]


def is_duplicated(text):
    text = re.sub("[\(\)\[\]\{\}/:]", "", text)
    text = re.sub("-", " ", text)
    sentences = text.split("։ ")
    # sentence_set = set(sentences)
    duplicate_sentences = set([sentence for sentence, count in collections.Counter(sentences).items() if count > 1])
    print("duplicates", len(duplicate_sentences))
    return len(duplicate_sentences)


def remove_duplicate_sentences(text):
    """
    Given a text, remove sentences which are duplicated.
    :param text: a piece of text
    :return: text without duplicate sentences
    """
    text = re.sub("[\(\)\[\]\{\}/:]", "", text)
    text = re.sub("-", " ", text)
    sentences = sentence_tokenizer(text)
    #sentence_set = set(sentences)
    duplicate_sentences = set([sentence for sentence, count in collections.Counter(sentences).items() if count > 1])
    print("duplicates", len(duplicate_sentences))
    for s in duplicate_sentences:
        sentence = s.strip()
        try:
            if len(sentence)>10: # len(re.findall("[\(\)\[\]\{\}-]", sentence)) == 0 and
                match_objects = list(re.finditer(sentence, text))
                if len(match_objects) >1 and match_objects is not None:
                    replace_span = [match_objects[i].span() for i in range(1, len(match_objects))]
                    for span in reversed(replace_span):
                        start = span[0]
                        end = span[1]
                        text = text[:start] + text[end+1:]
        except re.error:
            pass
            #print("Some character cannot be compiled in the following sentence:\n", sentence)
    return text


def group_texts(text_list, n):
    """
    Given a list of texts, returns a list of items concatenated n at a time.
    :param text_list: a list of text
    :param n: number of texts to concatenate
    :return: a new list of texts
    """
    start = 0
    end = 0
    if n > len(text_list):
        return text_list
    groups = []
    for i in range(0, ceil(text_list.__len__()/n)):
        end = start + n
        cat_text = ""
        for j in range(start, end):
            if j < text_list.__len__():
                cat_text = cat_text + text_list[j] + "\n\n"
        groups.append(cat_text)
        start = end
    return groups

