import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef, \
    f1_score, log_loss, roc_auc_score, roc_curve, balanced_accuracy_score

logger = logging.getLogger(__name__)


class ModelEvaluation:
    def __init__(self, estimator):
        self.estimator = estimator
        self.thresholds = [.2, .3, .4, .5, .6, .7, .8, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99]

    def model_validation(self, y_test, y_pred, thresholds=None) -> pd.DataFrame:
        """
        Model validation
        :param y_test:
        :param y_pred: predicted values
        :param thresholds: an array or series of classification thresholds
        :return:
        """
        # calculate AUC
        auc_score = round(roc_auc_score(y_test, y_pred) * 100, 2)
        logger.debug("AUC on the test set is {}%".format(auc_score))

        if thresholds is None:
            thresholds = self.thresholds

        # initialise a dictionary list to store all the metrics
        d_list = []
        for threshold in thresholds:
            predicted_output = [
                True if item > threshold else False for item in y_pred
            ]
            accuracy = round(accuracy_score(y_test, predicted_output) * 100, 2)
            balanced_accuracy = round(balanced_accuracy_score(y_test, predicted_output) * 100, 2)
            recall = round(recall_score(y_test, predicted_output) * 100, 2)
            precision = round(precision_score(y_test, predicted_output) * 100, 2)
            mcc = round(matthews_corrcoef(y_test, predicted_output) * 100, 2)
            f1 = round(f1_score(y_test, predicted_output) * 100, 2)
            ll = round(log_loss(y_test, predicted_output) * 100, 2)
            metric_dictionary = {
                "classification threshold": threshold,
                "auc": auc_score,
                "accuracy": accuracy,
                "balanced accuracy": balanced_accuracy,
                "recall": recall,
                "precision": precision,
                "f1 score": f1,
                "mcc": mcc,
                "log loss": ll,
            }
            d_list.append(metric_dictionary)
        metric_df = pd.json_normalize(d_list)
        logger.debug("Metrics on the test set")
        logger.debug("\n%s", metric_df.to_string(index=False))
        return metric_df

    def create_confusion_matrix_df(
            self,
            x_test: pd.DataFrame,
            y_test: pd.Series,
            cls_thresholds=None,
    ) -> pd.DataFrame:
        """
        Create a confusion matrix dataframe using different thresholds
        :param x_test:
        :param y_test:
        :param cls_thresholds: Array of classification thresholds
        :return: confusion matrix dataframe
        """
        if cls_thresholds is None:
            cls_thresholds = self.thresholds
        y_pred_prob = self.estimator.predict_proba(x_test)[:, 1]
        n = len(y_test)
        dict_list = []
        for threshold in cls_thresholds:
            y_pred = [True if p > threshold else False for p in y_pred_prob]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            dict_list.append(
                {
                    "threshold": threshold,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "TP(%)": round(tp / n * 100, 2),
                    "FP(%)": round(fp / n * 100, 2),
                    "TN(%)": round(tn / n * 100, 2),
                    "FN(%)": round(fn / n * 100, 2),
                    "Accuracy": round((tp + tn) / n * 100, 2),
                    "Recall": round(tp / (tp + fn) * 100, 2),
                    "Precision": round(tp / (tp + fp) * 100, 2),
                    "Predicted positives": sum(y_pred),
                    "Traffic Blocked(%)": round(sum(y_pred) / len(y_pred) * 100, 2),
                }
            )

        return pd.json_normalize(dict_list)

    @staticmethod
    def plot_cfm_bar_chart(df: pd.DataFrame, save_path: str = None):
        """
        Create a confusion matrix stacked bar chart from the confusion matrix dataframe returned
        by the create_confusion_matrix_df function
        :param save_path:
        :param df: output of the create_confusion_matrix_df function
        """
        # Set the figure size and create subplots
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get the number of classification thresholds
        num_thresholds = len(df)

        # Define the colors for true positives (TP), false positives (FP),
        # true negatives (TN), and false negatives (FN)
        colors = ['green', 'orange', 'blue', 'red']

        x = np.arange(num_thresholds)

        # Plot each confusion matrix as stacked bars
        for i in range(num_thresholds):
            tp = df['TP'][i]
            fp = df['FP'][i]
            tn = df['TN'][i]
            fn = df['FN'][i]

            # Plot the bars for each category
            ax.bar(x[i], tp, color=colors[0])
            ax.bar(x[i], fp, bottom=tp, color=colors[1])
            ax.bar(x[i], tn, bottom=tp + fp, color=colors[2])
            ax.bar(x[i], fn, bottom=tp + fp + tn, color=colors[3])

        # Set the x-axis tick labels to the classification thresholds
        ax.set_xticks(x)
        ax.set_xticklabels(df['threshold'], rotation=45)

        # Set the x-axis label
        ax.set_xlabel('Classification Threshold')

        # Set the y-axis label
        ax.set_ylabel('Count')

        # Set the title
        ax.set_title('Confusion Matrix for Different Classification Thresholds')

        # Set the legend
        ax.legend(['True Positives', 'False Positives', 'True Negatives', 'False Negatives'], loc='center left',
                  bbox_to_anchor=(1, 0.5))

        # Set the x-axis limits based on the number of thresholds
        ax.set_xlim(-0.5, num_thresholds - 0.5)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.tight_layout()
            plt.show()

    def create_plot_roc_curve(self, x_test: pd.DataFrame, y_true: pd.Series):
        """
        Plot the ROC curve. Input is either x_test or x_val
        :param x_test:
        :param y_true:
        """
        y_pred_prob = self.estimator.predict_proba(x_test)[:, 1]
        plot_roc_curve(y_true, y_pred_prob)
        plt.show()

    def calculate_classification_threshold(self, x_test, y_test, risk_appetite=0.02):
        """
        Given a risk appetite, calculate the classification threshold
        :param x_test:
        :param y_test:
        :param risk_appetite: percentage of predicted
        :return:
        """
        # Make predictions on the test data
        y_pred_proba = self.estimator.predict_proba(x_test)[:, 1]
        n = len(y_test)
        allowed_positives = round(n * risk_appetite, 0)
        # Calculate the FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Find the threshold that corresponds to allowing x rows to be predicted as 1
        threshold_idx = np.argmax(tpr >= allowed_positives / len(y_test))
        classification_threshold = thresholds[threshold_idx]
        return classification_threshold
