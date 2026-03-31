import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.viz.plot import Plot


class CatPlot(Plot):
    """
    Class to plot categorical data
    """
    def __init__(self):
        super().__init__()

    def barplot_cat_against_num(self, df: pd.DataFrame, cat_feature: str, num_feature: str, top_n: int = 10,
                                **kwargs):
        """
        Plot a bar plot for a categorical feature against a numerical feature.
        For example: plot the top 10 countries by income
        :param df: aggregated dataframe (often by value_count or groupby function)
        :param cat_feature: categorical feature
        :param num_feature: numerical feature
        :param top_n: number of top categories
        :param kwargs: other arguments to pass to the bar plot, e.g. color
        """
        if top_n == 0:
            top_n = df.shape[0]
        top_df = df.sort_values(by=num_feature, ascending=False).head(top_n)
        sns.set_theme(style="whitegrid", palette=self.palette)
        bars = plt.barh(top_df[cat_feature], top_df[num_feature], **kwargs)
        plt.xlabel(num_feature)
        plt.ylabel('')
        plt.title(f'Top {top_df.shape[0]} {cat_feature} by {num_feature}')
        plt.xticks(rotation=90)
        for bar, value in zip(bars, top_df[num_feature]):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{round(value, 2)}',
                     va='center', ha='left', fontsize=10)
        plt.show()

    def countplot(self, df: pd.DataFrame, feature: str):
        """
       Plot a count plot for a single categorical feature in a DataFrame.
       The plot contains 2 subplots, one for the count and one for the percentage
       :param df: DataFrame
       :param feature: name of the categorical feature
       """
        sns.set_theme(style="whitegrid", palette=self.palette)
        _, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Bar plot for the count
        df[feature].value_counts().plot(kind='barh', ax=axes[0])
        axes[0].set(xlabel="", ylabel="Count", title=feature)
        for c in axes[0].containers:
            # set the bar label
            axes[0].bar_label(c, label_type="edge")

        # Pie plot for the percentage
        df[feature].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1], startangle=90)
        axes[1].set(xlabel="", ylabel="Percentage", title=feature)
        plt.show()

    def multiple_countplot(self, df: pd.DataFrame, cat_features=None):
        """
           Count values per feature in a series of rows and columns. This is for categorical features
           :param df: dataframe
           :param cat_features: columns to be plotted. If None, all cat columns will be plotted
           """
        if not cat_features:
            cat_features = df.select_dtypes(["object", "category"]).columns

        # Create the count plot
        num_rows, num_cols = self.find_num_rows_cols(len(cat_features))
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()
        for i, col in enumerate(cat_features):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid", palette=self.palette)
            df[col].value_counts().plot(kind="pie")
            ax.set(xlabel="", ylabel="", title=col)
        plt.show()

    def xtab(self, df: pd.DataFrame, target: str, cat_feature: str, option: str = "stacked", bar_label: bool = True):
        """
        Plot the cross-tab stacked-bar plot between 2 features, the target and categorical feature
        :param df:
        :param cat_feature:
        :param target:
        :param bar_label: by default, bar labels are shown
        :param option: "stacked" or "grouped"
        :return:
        """
        # Fill the missing values as NA in the categorical feature
        if df[cat_feature].dtype == "object":
            df[cat_feature] = df[cat_feature].fillna("NA")

        # Create the stacked bar plot
        if option == "stacked":
            ct = pd.crosstab(df[cat_feature], df[target])
            ax = ct.plot(kind="bar", stacked=True, rot=0)
            sns.set_theme(style="whitegrid", palette=self.palette)
            ax.legend(title=target, bbox_to_anchor=(1, 1.02), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
        # Create the grouped bar plot
        elif option == "grouped":
            ax = sns.countplot(
                data=df, x=cat_feature, hue=target, order=df[cat_feature].unique(),
                palette=self.palette
            )
            sns.move_legend(ax, bbox_to_anchor=(1, 1.02), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
        # The option is not valid
        else:
            raise ValueError("Wrong value for option. Only accept 'stacked' or 'grouped'")

        # add annotations if desired
        if bar_label:
            for c in ax.containers:
                # set the bar label
                ax.bar_label(c, label_type="center")
        plt.show()

    def multiple_xtab(self, df: pd.DataFrame, target: str, features: list = None, bar_label: bool = True):
        """
        Plot multiple cross-tab grouped bars for 2 features, a target and a feature in a list of features
        :param df:
        :param features: list of features to be visualised as in the cross-tab plots
        :param target:
        :param bar_label: by default, bar labels are shown
        """
        if not features:
            features = df.select_dtypes(["object", "category"]).columns
        num_rows, num_cols = self.find_num_rows_cols(len(features))

        # Create the figure
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()

        # Get unique values in the target variable
        unique_values = df[target].unique()

        # Create a colormap
        colormap = plt.get_cmap(self.palette, len(unique_values)).colors

        # Create a color palette
        color_palette = dict(zip(unique_values, colormap))

        for i, col in enumerate(features):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid")
            sns.countplot(data=df, x=col, hue=target, ax=ax, palette=color_palette)
            sns.move_legend(ax, bbox_to_anchor=(1, 1.02), loc="upper left")
            ax.set(xlabel="", ylabel="", title=col)
            ax.tick_params(axis="x", rotation=45)
            # add annotations if desired
            if bar_label:
                for c in ax.containers:
                    # set the bar label
                    ax.bar_label(c, label_type="center")
        plt.show()

    def pairplots_against_target(self, df: pd.DataFrame, target: str):
        """
        This is to show all the pair-plots between any two data features
        with the cat_feature being highlighted. For example, the correlation
        between Age and Income with the highlight as Gender
        Note: this function is only good for a small number of features (smaller than 5)
        :param df:
        :param target:
        """
        # Drop the target column
        df = df.drop([target], axis=1)
        # Pair-plot with all the categorical variables inside the dataframe
        sns.pairplot(df, hue=target, aspect=1.5, palette=self.palette)
        plt.show()

    def multiple_scatter_against_target(self, df: pd.DataFrame, target: str, cat_features=None):
        """
        Correlation between all categorical variables against the target
        :param cat_features: categorical variables
        :param df:
        :param target: target variable name
        """
        if cat_features is None:
            cat_features = df.select_dtypes(exclude="number").drop(columns=target).columns
        num_rows, num_cols = self.find_num_rows_cols(len(cat_features))

        # Create a subplot
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20 * num_rows / num_cols))
        axes1d = axes.ravel()
        for i, col in enumerate(cat_features):
            ax = axes1d[i]
            sns.set_theme(style="whitegrid")
            sns.catplot(x=col, y=target, data=df, ax=ax, palette=self.palette)
            ax.set(xlabel="", ylabel="", title=col)
        plt.show()
