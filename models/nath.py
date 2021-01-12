from typing import List, Tuple
from .nath_et_al.src.algorithms.threshold_clustering.executor import execute_threshold_clustering
import preprocess_NLP_pkg


class NathModel(object):

    def __init__(self, hyperparams, features):
        pass

    def test(self, test_set: List[Tuple[str, dict]]):
        docs = [x for x, y in test_set]
        pred_results = self.analyse_documents(docs)
        return pred_results

    def analyse_documents(self, documents: List[str]):
        results = []
        for document in documents:
            results.append(self._analyse(document))
        return results

    def _analyse(self, document: str):
        prediction_TBC = execute_threshold_clustering(document, merge_threshold=50, add_node_threshold=50,
                                                      prune=True, number_of_terms=50,
                                                      distance_measure=preprocess_NLP_pkg.clark_distance,
                                                      use_duplication_feature=False)
        return {
            "style_change": bool(prediction_TBC > 1)
        }
