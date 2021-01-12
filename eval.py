# -*- coding: utf-8 -*-
import sys
import os
import json
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from get_texts_from_path import get_docs, get_vecs
from models.utils import evaluation17
from models import ACModel, NathModel,  ZlatkovaModel, KarasModel
from feature_extraction.computation import feature_extractors
from save_predictions import save_predictions, save_vectors

"""Configuration example:
```
{
  "tasks":["style_change", "style_breach"],
  "model": {
    "name": "karas",  # other options: "zlatkova", "nath", "pnb", "lof", "ac"
    "hyperparams": {
      ...  # list of hyperparams and their values
    },
    "trainable": false, # true if model is trainable
  }
  "use_vectors": false, # true if have collected document vectors
  "features": {
    "extractors": [],  # if not empty, should specify feature groups
    "selection": false,  # if true, performs feature selection
    "pca": false  # if true, performs dimensionality reduction using PCA
  },
  "datasets": {
    "train": "",  # path to train dataset
    "dev": "",  # path to development dataset
    "test": []  # paths to test datasets
  },
  "outputDir": "", # path to save predictions,
  "resultDir": "" # path to save metrics,
  "vectorsDir": "" # if use_vectors is False its a path to save a folder with computed vectors  in 
                     else a path that contain that filled folder
}
```
"""

methods = {
    "zlatkova": ZlatkovaModel,
    "nath": NathModel,
    "karas": KarasModel,
    "ac": ACModel
}

def feature_selection(features):
    selector = VarianceThreshold()  # selects all features with non-zero variance
    return lambda x: selector.fit_transform(features(x))


def pca(features):
    reducer = PCA()  # picks all principal components
    return lambda x: reducer.fit_transform(features(x))


def process_results(test, task, name, results):
    print("test data: {},\ntask: {},\nmodel: {},\nprecision = {}, recall = {}, f1-score = {}, accuracy = {}".
          format(test, task, name, results[0], results[1], results[2], results[3]))


def eval(config):
    features = feature_extractors.keys()
    if config["features"]["selection"]:
        features = [feature_selection(feature) for feature in features]
    if config["features"]["pca"]:
        features = pca(features)

    method = methods[config["model"]["name"]]
    model = method(config["model"]["hyperparams"], features)
    if config["model"].get("trainable", False):
        model.train(get_docs(config["datasets"]["train"]), get_docs(config["datasets"]["dev"]))

    for test_dir in config["datasets"]["test"]:
        docs = get_docs(path=test_dir)
        texts = [(text, dictionary) for _, text, dictionary in docs]

        if config["model"]["name"] in ["ac", "karas"]:
            folder_name = "-".join(
                [os.path.split(test_dir)[-1], str(config["features"]["selection"]), str(config["features"]["pca"])])
            use_vectors = config["use_vectors"]
            if config["use_vectors"]:
                vecs_for_algorithm = get_vecs(path=test_dir, vecpath=os.path.join(config["vectorsDir"], folder_name))
                texts = [(text, vector, dictionary) for _, text, vector, dictionary in vecs_for_algorithm]
            predictions_and_vectors = model.test(texts, use_vectors)
            predictions = predictions_and_vectors[0]
            vectors = predictions_and_vectors[1]
            if not use_vectors:
                save_vectors(docs, vectors, config, folder_name)
        else:
            predictions = model.test(texts)

    save_predictions(docs, predictions, config, os.path.split(test_dir)[-1])
    results = evaluation17.main(docs, predictions, "style_breach" == config["task"], config,
                                os.path.split(test_dir)[-1])
    process_results(test_dir, config["task"], config["model"]["name"], results)


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf8") as config_json:
        config = json.load(config_json)
    eval(config)
