from typing import List, Tuple
from .zlatkova_et_al.main import train
from .zlatkova_et_al.load_model import load_model


class ZlatkovaModel(object):

    def __init__(self, model_path=None):
        self.model_path = model_path

    def train(self, train_set: List[Tuple[str, dict]], dev_set: List[Tuple[str, dict]]):
        train(train_set, dev_set, path_to_save_model=self.model_path)

    def test(self, test_set: List[Tuple[str, dict]]):
        if self.model_path is None:
            raise IOError("No trained model to make predictions")
        docs = [x for x, y in test_set]
        pred_results = self.analyse_documents(docs)
        return sum([test[1]["style_change"] == pred["style_change"] for test, pred in zip(test_set, pred_results)]) / len(test_set)

    def analyse_documents(self, documents: List[str]):
        results = []
        model = load_model(self.model_path)
        for document in documents:
            results.append(self._analyse(model, document))
        return results

    def _analyse(self, model, document: str):
        prediction = model.predict([document])[0]
        return {
            "style_change": prediction
        }
