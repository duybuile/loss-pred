import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class FeatureStabilityFilter:
    """Feature Stability filter based on train/test split
    (Ref:  https://www.youtube.com/watch?v=W1ALky-wHBU)"""

    def __init__(self, auc_threshold=0.7, sample_rate=0.5, seed=42):
        self.sample_rate = sample_rate
        self.random_state = seed
        self.auc_threshold = auc_threshold
        self.auc_series = pd.Series()
        self.columns_to_drop = []

    def fit(self, X_train, X_test):
        train_sample = X_train.sample(frac=self.sample_rate, random_state=self.random_state)
        train_sample['target'] = 0

        train_sample_size_ratio = (train_sample.shape[0] / X_test.shape[0])
        if train_sample_size_ratio > 1:
            train_sample_size_ratio = 1

        test_sample = X_test.sample(frac=train_sample_size_ratio, random_state=self.random_state)
        test_sample['target'] = 1

        feat_stab_df = train_sample.append(test_sample)
        feat_stab_df = feat_stab_df.sample(frac=1, random_state=self.random_state)

        auc_dict = {}
        X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
            feat_stab_df.drop('target', axis=1), feat_stab_df['target'],
            test_size=0.25,
            shuffle=True,
            stratify=feat_stab_df['target'],
            random_state=42
        )

        for c in tqdm(X_train_fs.columns):
            dt = DecisionTreeClassifier(random_state=self.random_state)
            dt.fit(X_train_fs[c].values.reshape(-1, 1), y_train_fs.values)
            y_pred = dt.predict(X_test_fs[c].values.reshape(-1, 1))
            if y_pred.mean() == 0:
                continue
            if y_pred.mean() == 1.0:
                auc_dict[c] = 1.0
                continue
            auc = roc_auc_score(y_pred, y_test_fs)
            auc_dict[c] = auc

        aud_df = pd.Series(auc_dict)
        self.auc_series = aud_df.sort_values(ascending=False)
        self.columns_to_drop = list(self.auc_series.where(lambda x: x > self.auc_threshold).dropna().index)
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)
