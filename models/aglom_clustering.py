from typing import List, Tuple
from models.ac.main import make_prediction
from models.utils import text_segmentation
from feature_extraction.computation import compute_feature_vectors
import time
import datetime


class ACModel(object):

    def __init__(self, hyperparams: List, features: List[str]):
        self.features = features
        self.hyperparams = hyperparams

    def train(self, train_set: List[Tuple[str, dict]], dev_set: List[Tuple[str, dict]]):
        pass

    def test(self, test_set: List[Tuple[str, dict]], use_vectors):
        if use_vectors:
            docs_vectors = [(t, v) for t, v, d in test_set]
            pred_results = self.analyse_documents(docs_vectors, use_vectors)
        else:
            docs = [x for x, y in test_set]
            pred_results = self.analyse_documents(docs, use_vectors)
        dict = pred_results[0]
        vectors = pred_results[1]
        return dict, vectors

    def analyse_documents(self, documents: List[str], use_vectors):
        results = []
        i = 1
        l = len(documents)
        if use_vectors:
            docs_vectors = documents
            for text, paragraph_vectors in docs_vectors:
                print("working on the", i, "/", l, datetime.datetime.now().time())
                start_time = time.time()
                paragraphs = text_segmentation.get_paragraphs_of(text)
                results.append(self._analyse(paragraph_vectors, paragraphs))
                print("computation time:", time.time() - start_time)
                i += 1
        else:
            docs_vectors = []
            for document in documents:
                print("working on the", i, "/", l, datetime.datetime.now().time())
                start_time = time.time()
                paragraphs = text_segmentation.get_paragraphs_of(document)
                if self.features:
                    feature_vectors = compute_feature_vectors(document, paragraphs, self.features)
                else:
                    feature_vectors = []
                docs_vectors.append(feature_vectors)
                results.append(self._analyse(feature_vectors, paragraphs))
                print("computation time:", time.time()-start_time)
                i += 1
        return results, docs_vectors

    def _analyse(self, feature_vectors: List[List[float]], paragraphs: List[str]):
        predicted = make_prediction(feature_vectors, paragraphs, self.hyperparams)
        return {
            "style_change": predicted["style_change"],
            "style_breaches": predicted["style_breaches"]
        }
