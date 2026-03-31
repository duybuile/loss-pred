import joblib
import pandas as pd


class ModelServing:
    def __init__(self, estimator: object, estimator_path: str):
        if estimator is not None:
            self.estimator = estimator
        else:
            self.estimator = self.load_estimator(estimator_path)

    @staticmethod
    def load_estimator(path: str):
        """
        Load the estimator/classifier from a path
        :param path:
        :return:
        """
        return joblib.load(path)

    def model_serving(self, x: pd.DataFrame, classification_threshold: float):
        """
        Make prediction on the new dataset
        :param x:
        :param classification_threshold:
        :return:
        """
        x["pred_prob"] = self.estimator.predict_proba(x)[:, 1]
        x["pred"] = [
            True if p > classification_threshold else False for p in x["pred_prob"]
        ]
        return x
