import os
import json
import numpy as np
import re


def read_json(json_doc):
    with open(json_doc, "r", encoding="utf8") as src:
        doc = json.load(src)
    return doc


def get_docs_batch(path, start, end):
    docs = []
    files = sorted(os.listdir(path))
    batch = files[start:end]
    for json_doc in batch:
        extracted_doc = read_json(os.path.join(path, json_doc))
        sources = np.array(extracted_doc['paragraph_source_docs'])
        indices = np.where(sources[:-1] != sources[1:])[0]
        indices = indices + np.ones(len(indices), dtype=int)
        breaches = []
        changes = False
        if len(indices):
            text = []
            changes = True
            indices = np.insert(indices, 0, 0)
            indices = np.insert(indices, len(indices), len(sources))
            length = 0
            for i in range(len(indices) - 1):
                if i == len(indices) - 2:
                    text.append('\n'.join(extracted_doc['paragraphs'][indices[i]:indices[i + 1]]))
                else:
                    text.append('\n'.join([*extracted_doc['paragraphs'][indices[i]:indices[i + 1]], '']))
                    length += len(text[-1])
                    breaches.append(length)
            text = ''.join(text)
        else:
            text = '\n'.join(extracted_doc['paragraphs'])
        if len(text) > 0:
            docs.append((json_doc, text, {"style_change": changes, "style_breaches": breaches}))
    return docs


def get_docs(path):
    docs = []
    files = sorted(os.listdir(path))
    for json_doc in files:
        extracted_doc = read_json(os.path.join(path, json_doc))
        sources = np.array(extracted_doc['paragraph_source_docs'])
        indices = np.where(sources[:-1] != sources[1:])[0]
        indices = indices + np.ones(len(indices), dtype=int)
        breaches = []
        changes = False
        if len(indices):
            text = []
            changes = True
            indices = np.insert(indices, 0, 0)
            indices = np.insert(indices, len(indices), len(sources))
            length = 0
            for i in range(len(indices) - 1):
                if i == len(indices) - 2:
                    text.append('\n'.join(extracted_doc['paragraphs'][indices[i]:indices[i + 1]]))
                else:
                    text.append('\n'.join([*extracted_doc['paragraphs'][indices[i]:indices[i + 1]], '']))
                    length += len(text[-1])
                    breaches.append(length)
            text = ''.join(text)
        else:
            text = '\n'.join(extracted_doc['paragraphs'])
        if len(text) > 0:
            docs.append((json_doc, text, {"style_change": changes, "style_breaches": breaches}))
    return docs


def get_vecs(path, vecpath):
    doc_vecs = []
    files = sorted(os.listdir(path))
    filenames = [re.sub('json$', '', file)for file in files]
    for fn in filenames:
        extracted_vec = read_json(os.path.join(vecpath, fn + 'truth'))["vector"]
        extracted_doc = read_json(os.path.join(path, fn + 'json'))
        sources = np.array(extracted_doc['paragraph_source_docs'])
        indices = np.where(sources[:-1] != sources[1:])[0]
        indices = indices + np.ones(len(indices), dtype=int)
        breaches = []
        changes = False
        if len(indices):
            text = []
            changes = True
            indices = np.insert(indices, 0, 0)
            indices = np.insert(indices, len(indices), len(sources))
            length = 0
            for i in range(len(indices) - 1):
                if i == len(indices) - 2:
                    text.append('\n'.join(extracted_doc['paragraphs'][indices[i]:indices[i + 1]]))
                else:
                    text.append('\n'.join([*extracted_doc['paragraphs'][indices[i]:indices[i + 1]], '']))
                    length += len(text[-1])
                    breaches.append(length)
            text = ''.join(text)
        else:
            text = '\n'.join(extracted_doc['paragraphs'])
        if len(text) > 0:
            doc_vecs.append((fn, text, extracted_vec, {"style_change": changes, "style_breaches": breaches}))
    return doc_vecs

