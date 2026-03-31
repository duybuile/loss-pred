import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features with non-zero coefficients using LASSO model.
    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> pipeline = Pipeline(steps=[('lasso_feature_selection', LassoFeatureSelector(cv=5, max_iter=20000, random_state=42))])
        >>> # Fit the pipeline on the training data
        >>> pipeline.fit(X_train, y_train)
        >>> # Transform both training and testing data
        >>> X_train_selected = pipeline.transform(X_train)
        >>> X_test_selected = pipeline.transform(X_test)
    """
    def __init__(self, cv=5, max_iter=20000, alphas=None, random_state=42):
        self.cv = cv
        self.max_iter = max_iter
        self.alphas = alphas
        self.random_state = random_state
        self.model = None
        self.selected_columns = None

    def fit(self, X, y):
        # Standardize the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit the LASSO model
        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, alphas=self.alphas, random_state=self.random_state)
        lasso.fit(X_scaled, y)

        # Select features with non-zero coefficients
        self.model = SelectFromModel(lasso, prefit=True)
        self.selected_columns = X.columns[self.model.get_support()]

        return self

    def transform(self, X):
        # Standardize the data
        X_scaled = self.scaler.transform(X)

        # Select the same features on the new data
        X_selected = self.model.transform(X_scaled)

        # Return as a DataFrame with the selected columns
        return pd.DataFrame(X_selected, columns=self.selected_columns, index=X.index)
