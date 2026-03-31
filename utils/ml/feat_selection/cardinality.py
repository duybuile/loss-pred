"""
Feature selection using low or high cardinality
"""
from sklearn.base import TransformerMixin
from tqdm import tqdm


class CardinalityFilterTransformer(TransformerMixin):
    """Removes columns based on low or high cardinality"""

    def __init__(self, high_cardinality_th=0.9):
        self.high_cardinality_th = high_cardinality_th
        self.cols_with_cardinality_one = []
        self.cols_with_high_cardinality = []

    def fit(self, x):
        nunique = x.nunique()
        col_cardinality = (nunique / x.shape[0]).sort_values(ascending=False)
        self.cols_with_high_cardinality = list(
            col_cardinality[col_cardinality > self.high_cardinality_th].index
        )
        # for cardinality one
        self.cols_with_cardinality_one = list(nunique[nunique == 1].index)
        return self

    def transform(self, x):
        return x.drop(
            self.cols_with_high_cardinality + self.cols_with_cardinality_one, axis=1
        )


class LowerCardinalityTransformer(TransformerMixin):
    """Lowers the cardinality of high cardinality columns"""

    def __init__(self, max_categories=10, others_category="others"):
        self.max_categories = max_categories
        self.others_category = others_category
        self.cols_with_high_cardinality = []
        self.category_mapping = []

    def fit(self, x):
        cat_cols = x.select_dtypes(include=["object", "string"]).columns
        cat_count = x[cat_cols].nunique()
        self.cols_with_high_cardinality = list(
            cat_count[cat_count > self.max_categories].index
        )

        for c in tqdm(self.cols_with_high_cardinality, desc="Fit: "):
            # find top X categories
            top_x = list(x[c].value_counts().head(10).index)
            self.category_mapping.append({"category": c, "top_X": top_x})
        return self

    def transform(self, x):
        # replace all others with 'others_category'
        for lv in tqdm(self.category_mapping, desc="Transform: "):
            c = lv["category"]
            top_x = lv["top_X"]
            x[c][~x[c].isin(top_x)] = self.others_category
        return x
