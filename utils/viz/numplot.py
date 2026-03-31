import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.viz.plot import Plot


class NumPlot(Plot):
    """
    Class to plot numeric data
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def heatmap(df: pd.DataFrame):
        """
        Correlation between all the numeric variables
        :param df:
        :return: correlation matrix in a dataframe format
        """
        num_df = df.select_dtypes("number")
        correlations = num_df.sort_index(axis=1).corr()
        _, ax = plt.subplots(figsize=(20, 20))
        # Create a heatmap of the correlation matrix, round the value to 2 decimals
        sns.heatmap(correlations.round(2), ax=ax, annot=True, vmin=-1, vmax=1, cmap="seismic_r")
        plt.show()

    def single_scatter_overtime(self, df: pd.DataFrame, num_field: str, timestamp_field: str, target: str):
        """
        Scatter plot over time for a numerical field against the target
        :param df:
        :param num_field: numerical field
        :param timestamp_field: timestamp field
        :param target: target
        :return:
        """
        ax = sns.scatterplot(data=df, x=timestamp_field, y=num_field, hue=target)
        sns.set_theme(style="whitegrid", palette=self.palette)
        sns.move_legend(ax, bbox_to_anchor=(1, 1.02), loc="upper left")
        # Set x-axis label to empty
        ax.set(xlabel="")
        # Rotate x-axis labels to 45 degrees for better visibility
        ax.tick_params(axis="x", rotation=45)
        plt.show()

    def multiple_scatter_overtime(self, df: pd.DataFrame, target: str, timestamp_field: str,
                                  num_fields: list = None):
        """
        Multiple scatter plot over time for numerical fields
        :param df:
        :param num_fields: list of numerical fields
        :param timestamp_field: timestamp field
        :param target: target
        """
        # If num_fields is None, use all numerical fields from df
        if num_fields is None:
            num_fields = df.select_dtypes(["number"]).columns
        # Calculate the number of rows for the subplot
        num_rows, num_cols = self.find_num_rows_cols(len(num_fields))
        # Create a subplot
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()
        for i, col in enumerate(num_fields):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid", palette=self.palette)
            sns.scatterplot(data=df, x=timestamp_field, y=col, hue=target, ax=ax)
            # Set x-axis and y-axis labels to empty
            ax.set(xlabel="", ylabel="", title=col)
            # Rotate x-axis labels to 45 degrees for better visibility
            ax.tick_params(axis="x", rotation=45)
        plt.show()

    def multiple_boxplot(self, df: pd.DataFrame, features: list = None):
        """
        Boxplot for all the numeric variables in a dataframe
        :param df: dataframe
        :param features: numeric variables to be plotted. If None, all numeric variables will be plotted
        :return:
        """
        if features is None:
            features = df.select_dtypes("number").columns
        num_rows, num_cols = self.find_num_rows_cols(len(features))
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()
        for i, col in enumerate(features):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid", palette=self.palette)
            sns.boxplot(y=col, data=df, ax=ax)
            ax.set(xlabel="", ylabel="", title=col)
        plt.show()

    def quantile_plot(self, df: pd.DataFrame, num_field: str, target: str, quantile=None, plot=True):
        """
        Look at quantile of a numeric field over a categorical field
        :param df:
        :param num_field:
        :param target:
        :param quantile:
        :param plot: if True, show the plot. Otherwise, show the table
        :return:
        """
        # Create a quantile series when the quantile is not provided
        if quantile is None:
            quantile = np.arange(0.05, 0.95, 0.05)
        g = df.groupby(target)
        summary_df = g[num_field].quantile(quantile).unstack()
        if plot:
            summary_df.T.plot(subplots=True)
            sns.set_theme(style="whitegrid", palette=self.palette)
            plt.title(f"Quantile plot of {num_field} over {target}")
            plt.show()
            return plt
        else:
            return summary_df

    def histogram(self, df: pd.DataFrame, features: list = None):
        """
        Histogram for numeric variables from a list in a dataframe
        :param df:
        :param features: numeric features
        """
        if features is None:
            features = df.select_dtypes("number").columns
        num_df = df[features]
        sns.set_theme(style="whitegrid", palette=self.palette)
        num_df.hist(figsize=(20, 20))
        plt.show()

    def histogram_density(self, df: pd.DataFrame, features: list = None):
        """
        Histogram and density plot for numeric variables
        :param df:
        :param features: numeric columns
        """
        if features is None:
            features = df.select_dtypes("number").columns
        for col in features:
            _, ax = plt.subplots(1, 2, figsize=(15, 4))
            sns.set_palette(self.palette)
            sns.histplot(df, x=col, bins=30, ax=ax[0])
            ax[0].set_title(f"Histogram for {col}")
            sns.kdeplot(
                data=df, x=col, fill=True, common_norm=False, ax=ax[1]
            )
            ax[1].set_title(f"Density plot for {col}")
        plt.show()

    def histogram_density_against_target(self, df: pd.DataFrame, target: str, features: list = None):
        """
        Histogram and density plot for numeric variables against the target
        :param df:
        :param target: the categorical variable (could be the target variable)
        :param features: numeric columns
        """
        if features is None:
            features = df.select_dtypes("number").columns
        for col in features:
            _, ax = plt.subplots(1, 2, figsize=(15, 4))
            sns.set_palette(self.palette)
            sns.histplot(df, x=col, hue=target, bins=30, ax=ax[0])
            ax[0].set_title(f"Histogram for {col}")
            sns.kdeplot(
                data=df, x=col, hue=target, fill=True, common_norm=False, ax=ax[1]
            )
            ax[1].set_title(f"Density plot for {col}")
        plt.show()

    def plot_over_time(self, df: pd.DataFrame, time_feature: str, num_features: list = None,
                       function: str = "count", plot_type: str = "bar"):
        """
        Bar/Line plot over time
        Step 1: aggregate the numeric fields with the corresponding function
        Step 2: plot the bar/line plot
        :param df:
        :param time_feature: generally months or years
        :param num_features: numeric features
        :param function: count, mean, min, max
        :param plot_type: 'bar' or 'line'
        :return:
        """
        if num_features is None:
            num_features = df.select_dtypes("number").columns

        agg_df = df.groupby(time_feature)[num_features].agg(function).reset_index()
        # Create subplots
        num_rows, num_cols = self.find_num_rows_cols(len(num_features))
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()
        for i, col in enumerate(num_features):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid", palette=self.palette)
            # Plot a bar or a line chart
            if plot_type == "bar":
                agg_df.plot(x=time_feature, y=col, kind="bar", ax=ax)
            elif plot_type == "line":
                agg_df.plot(x=time_feature, y=col, kind="line", ax=ax)
            ax.set(xlabel="", ylabel="", title=col)
            ax.tick_params(axis="x", rotation=45)
        plt.show()
