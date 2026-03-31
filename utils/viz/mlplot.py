# This class is to plot any ML results
import json

from matplotlib import pyplot as plt

from utils.viz.plot import Plot


class MLPlot(Plot):
    def __init__(self, palette: str = 'Paired'):
        super().__init__(palette)

    @staticmethod
    def plot_lime_feature_importance(lime_output_json: str, top_n: int = 10):
        """
        Plots the top N most important features from LIME output with diverging bars.

        Parameters:
        lime_output_json (str): JSON string containing LIME feature importance.
        top_n (int): Number of top features to display. Default is 10.

        Examples:
        >>> lime_output_json = '{"feature_1 >= 0.2": -0.34, "feature_2 <= 34": 0.21, "feature_3 >= 3": 0.1}'
        >>> MLPlot.plot_lime_feature_importance(lime_output_json, top_n=2)
        """
        # Parse the JSON string into a dictionary
        lime_output = json.loads(lime_output_json)

        # Sort features by absolute importance score in descending order
        sorted_features = sorted(lime_output.items(), key=lambda x: abs(x[1]), reverse=True)

        # Select top N features
        top_features = sorted_features[:top_n]

        # Extract feature names and importance scores
        features = [x[0] for x in top_features]
        importance_scores = [x[1] for x in top_features]

        # Create a diverging bar chart
        plt.figure(figsize=(10, 6))
        colors = ['orange' if score < 0 else 'blue' for score in importance_scores]  # Color coding
        bars = plt.barh(features, importance_scores, color=colors, edgecolor='black')

        # Add importance values at the end of each bar
        for bar, score in zip(bars, importance_scores):
            if score < 0:
                plt.text(score - 0.02, bar.get_y() + bar.get_height() / 2, f'{score:.2f}',
                         va='center', ha='center', color='black')
            else:
                plt.text(score + 0.02, bar.get_y() + bar.get_height() / 2, f'{score:.2f}',
                         va='center', ha='center', color='black')

        # Add a vertical line at x=0 for reference
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

        # Labels and title
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} LIME Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
        plt.tight_layout()
        plt.show()
