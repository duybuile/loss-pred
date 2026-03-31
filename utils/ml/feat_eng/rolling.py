import pandas as pd
from codetiming import Timer


class RollingFeatures:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    @Timer(
        name="count of the last n days",
        text="Elapsed time for {name}: {milliseconds:.0f} ms",
    )
    def count_of_last_n_days(
        self, df: pd.DataFrame, identifier: str, delta: int
    ) -> pd.DataFrame:
        temp_df = (
            df.set_index(self.timestamp)
            .groupby(identifier, sort=False)[identifier]
            .rolling("%sd" % delta, closed="both")
            .count()
            .rename("count_%s_%sd" % (identifier, delta))
        )
        temp_df = temp_df[~temp_df.index.duplicated(keep="first")]
        return df.merge(
            temp_df, how="left", left_on=[identifier, self.timestamp], right_index=True
        )

    @Timer(
        name="sum of the last n days",
        text="Elapsed time for {name}: {milliseconds:.0f} ms",
    )
    def sum_of_last_n_days(
        self, df: pd.DataFrame, identifier: str, value: str, delta: int
    ) -> pd.DataFrame:
        temp_df = (
            df.set_index(self.timestamp)
            .groupby(identifier, sort=False)[value]
            .rolling("%sd" % delta, closed="both")
            .sum()
            .rename("sum_%s_%d" % (identifier, delta))
        )
        temp_df = temp_df[~temp_df.index.duplicated(keep="first")]
        return df.merge(
            temp_df, how="left", left_on=[identifier, self.timestamp], right_index=True
        )

    @Timer(
        name="mean of the last n days",
        text="Elapsed time for {name}: {milliseconds:.0f} ms",
    )
    def mean_of_last_n_days(
        self, df: pd.DataFrame, identifier: str, value: str, delta: int
    ) -> pd.DataFrame:
        temp_df = (
            df.set_index(self.timestamp)
            .groupby(identifier, sort=False)[value]
            .rolling("%sd" % delta, closed="both")
            .mean()
            .rename("mean_%s_%sd" % (identifier, delta))
        )
        temp_df = temp_df[~temp_df.index.duplicated(keep="first")]
        return df.merge(
            temp_df, how="left", left_on=[identifier, self.timestamp], right_index=True
        )

    def fit(
        self, df: pd.DataFrame, identifier: str, value: str, delta: int
    ) -> pd.DataFrame:
        df = self.count_of_last_n_days(df, identifier, delta)
        df = self.sum_of_last_n_days(df, identifier, value, delta)
        df = self.mean_of_last_n_days(df, identifier, value, delta)
        return df


class RollingWithGrace:
    def __init__(self, timestamp, grace=30, delta=150):
        self.timestamp = timestamp
        self.grace = grace
        self.delta = delta

    @Timer(
        name="Rolling sum with grace",
        text="Elapsed time for {name}: {milliseconds:.0f} ms",
    )
    def sum_of_first_n_days(
        self, df: pd.DataFrame, identifier: str, value_col: str
    ) -> pd.DataFrame:
        s1 = (
            df.groupby(identifier)
            .rolling("%sd" % self.delta, on=self.timestamp)[value_col]
            .sum()
            .rename("sum_%s_grace_%sd" % (identifier, self.grace))
        )
        s2 = (
            df.groupby(identifier)
            .rolling("%sd" % self.grace, on=self.timestamp)[value_col]
            .sum()
            .rename("sum_%s_grace_%sd" % (identifier, self.grace))
        )
        s = s1.sub(s2)
        s = s[~s.index.duplicated(keep="first")]
        return df.merge(
            s, how="left", left_on=[identifier, self.timestamp], right_index=True
        )


def rolling_target(df: pd.DataFrame, timestamp: str, target: str, rolling_features: list, rolling_days=150):
    """
    Rolling a PII feature for the past {rolling_days}. Calculate the total appearance of
    a PII feature for the past {rolling_days} that causes a target to be 1.
    :param df:
    :param timestamp: timestamp feature
    :param target: target variable
    :param rolling_features: PII features such as phone number, email, SSN
    :param rolling_days: number of days for the rolling (default: 150 days)
    """
    rolling_object = RollingFeatures(timestamp)
    for feat in rolling_features:
        df = rolling_object.sum_of_last_n_days(df, feat, target, rolling_days)
        df = df.rename(columns={(f"sum_{feat}_%s" % rolling_days): f"bad_{feat}"})
    return df


def rolling_related_features(df: pd.DataFrame, timestamp: str, feature: str, rolling_features: list, rolling_window=30):
    """
    Calculate the sum, mean and count of a {feature} against a PII for the past {rolling_window} days
    :param df:
    :param timestamp:
    :param feature:
    :param rolling_features:
    :param rolling_window: number of days for the rolling (default: 30 days)
    """
    rolling_object = RollingFeatures(timestamp)
    for feat in rolling_features:
        df = rolling_object.fit(df, feat, feature, rolling_window)
    return df


def rolling_with_grace(df: pd.DataFrame, timestamp: str, target: str, rolling_features: list, graces=None):
    """
    Rolling a PII feature for the past {rolling_days}. Calculate the total appearance of
    a PII feature for the past {rolling_days} that causes a target to be 1.
    :param df:
    :param timestamp: timestamp feature
    :param target:
    :param rolling_features:
    :param graces: by default (1, 7 , 30, 45 and 60 days).
    """
    if graces is None:
        graces = [1, 7, 30, 45, 60]
    for grace in graces:
        rolling_object = RollingWithGrace(timestamp, grace)
        for feat in rolling_features:
            df = rolling_object.sum_of_first_n_days(df, feat, target)
            df = df.rename(columns={"sum_%s_grace_%sd" % (feat, grace): "bad_%s_%sd" % (feat, grace)})
    return df
