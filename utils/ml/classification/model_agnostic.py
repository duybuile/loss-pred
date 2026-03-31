import dill
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt


class ModelAgnostic:
    def __init__(self, estimator):
        self.estimator = estimator

    def generate_shap_feature_importance(self, x_test: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        """
        Feature importance with SHAP
        :param x_test:
        :param save_path: path to save the plot
        :return: the SHAP value per column in a dataframe
        """
        explainer = shap.TreeExplainer(self.estimator)
        shap_values = explainer.shap_values(x_test)

        # Create a new figure and axis object for the SHAP summary plot
        fig, ax = plt.subplots()
        if save_path is not None:
            shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
            plt.savefig(save_path)
        else:
            shap.summary_plot(shap_values, x_test, plot_type="bar")
        return self._gen_shap_feature_importance_df(shap_values, x_test.columns)

    @staticmethod
    def _gen_shap_feature_importance_df(shap_values: list, columns: list) -> pd.DataFrame:
        # Aggregate SHAP values across classes and samples for each feature
        # Step 1: Take the mean absolute SHAP value across samples, for each class separately
        shap_mean_per_class = [np.abs(class_shap_values).mean(axis=0) for class_shap_values in shap_values]

        # Step 2: Average the SHAP values across classes to get a single importance value per feature
        shap_importance = np.mean(shap_mean_per_class, axis=0)

        # Create a DataFrame to store feature names and their corresponding importance
        importance_df = pd.DataFrame({
            "column_name": columns,
            "shap_importance": shap_importance
        })

        # Sort features by importance
        importance_df = importance_df.sort_values("shap_importance", ascending=False)
        return importance_df

    def generate_lime_explanations(self, explainer: LimeTabularExplainer, x_test: pd.DataFrame):
        """
        Generate LIME explanations for all data in X_test.
        Note: the function runs quite slowly for a large dataset.
        It's recommended to select only a few data points in X_test
        :param explainer: LIME explainer
        :param x_test:
        :return:
        """
        feature_names = x_test.columns
        explanations = []
        for i in range(len(x_test)):
            data_point = x_test.values[i]
            explanation = explainer.explain_instance(
                data_point,
                self.estimator.predict_proba,
                num_features=len(feature_names),
                top_labels=1
            )
            explanation_text = explanation.as_list(label=explanation.top_labels[0])
            explanations.append(explanation_text)
        return explanations

    def generate_lime_explanation(self, explainer: LimeTabularExplainer, x_test: pd.DataFrame, num_features: int = 10):
        """
        Generate LIME explanations for one data point in X_test.
        :param explainer: LIME explainer
        :param x_test:
        :param num_features: number of features to be explained. If set to 0, all features will be explained
        :return: the LIME explanation
        """
        if x_test.shape[0] > 1:
            raise ValueError("Only one data point can be explained at a time. Use generate_lime_explanations() instead")
        if num_features == 0:
            num_features = len(x_test.columns)
        explanation = explainer.explain_instance(
            x_test.values[0],
            self.estimator.predict_proba,
            num_features=num_features,
            top_labels=1
        )
        return explanation

    def create_lime_explainer(self, x_train):
        """
        Create LIME explainer
        :param x_train:
        :return:
        """
        feature_names = x_train.columns
        explainer = LimeTabularExplainer(
            training_data=x_train.values,
            feature_names=feature_names,
            class_names=self.estimator.classes_,
            mode="classification"
        )
        return explainer

    @staticmethod
    def save_lime_explainer(explainer, file_name):
        """
        Save LIME explainer using dill
        :param explainer:
        :param file_name: file_name should have extension *.ser (for serialised object)
        :return:
        """
        with open(file_name, 'wb') as file:
            dill.dump(explainer, file)

    @staticmethod
    def load_lime_explainer(file_name):
        """
        Load LIME explainer using dill
        :param file_name: file_name should have extension *.ser (for serialised object)
        :return:
        """
        with open(file_name, 'rb') as file:
            explainer = dill.load(file)
        return explainer
