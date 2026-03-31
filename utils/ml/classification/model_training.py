import logging

import joblib
import pandas as pd
import shap
from pycaret.classification import setup, compare_models, create_model

logger = logging.getLogger(__name__)


class ModelTraining:
    def __init__(self, seed: int = 100):
        self.random_seed = seed

    def train_with_pycaret(self, df: pd.DataFrame, target_col: str, **kwargs):
        """
        Train the model with pycaret
        :param df:
        :param target_col:
        :param kwargs: additional parameters for pycaret compare_model
        :return: best model (estimator)
        """
        setup(df, target=target_col, session_id=self.random_seed)
        best = compare_models(**kwargs)
        return best

    def create_custom_model(self, df: pd.DataFrame, target_col: str, model_name: str, **kwargs):
        """
        Create a custom model
        :param df:
        :param target_col: target column
        :param model_name: take the following values ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge',
        'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
        :param kwargs: additional parameters for create_model
        :return: estimator
        """
        setup(df, target=target_col, session_id=self.random_seed)
        model = create_model(model_name, **kwargs)
        return model

    @staticmethod
    def generate_shap(estimator: object, **kwargs):
        """
        Generate SHAP values
        :param estimator:
        :param kwargs:
        :return:
        """
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(**kwargs)
        return shap_values

    @staticmethod
    def dump_estimator(estimator: object, path: str):
        """
        Dump the estimator/classifier to a file (*.pkl)
        :param estimator:
        :param path:
        """
        logger.info("Dump an estimator to %s" % path)
        joblib.dump(estimator, path)
