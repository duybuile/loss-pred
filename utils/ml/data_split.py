import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, identifier: str, test_size: float = 0.3, random_state: int = 5):
    """
    Split data into train and test using an identifier. Note that the identifier does not need to be unique.
    For instance, we can have a dataset with company_id as identifier where each row contains a year of
    financial data for that company.
    :param df:
    :param identifier: name of the identifier
    :param test_size:
    :param random_state:
    :return:
    """
    train_size = 1 - test_size
    unique_list = df[identifier].unique()
    # Calculate the unique identifiers and their corresponding targets
    training_samp, test_samp = train_test_split(unique_list, train_size=train_size, test_size=test_size,
                                                random_state=random_state, shuffle=True)

    training_data = df[df[identifier].isin(training_samp)]
    test_data = df[df[identifier].isin(test_samp)]

    return training_data, test_data


def split_train_val_test(
    df: pd.DataFrame, target: str, test_size=0.1, val_size=0.1, seed=100
):
    """
    Split data into train, validation and test
    :param seed:
    :param df:
    :param target:
    :param test_size:
    :param val_size:
    :return:
    """
    train_size = 1 - test_size
    val_size = round(val_size / train_size, 2)
    x, x_test, y, y_test = split_train_test_with_target(df, target, test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, random_state=seed
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def split_train_test_with_target(
    df: pd.DataFrame, target: str, test_size=0.3, seed=100
):
    """
    Split data into training and testing with target
    :param seed:
    :param target:
    :param df:
    :param test_size:
    :return:
    """
    x = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )
    return x_train, x_test, y_train, y_test


def split_train_test_without_target(df: pd.DataFrame, test_size=0.3, seed=100):
    """
    Split data into training and testing without target
    :param seed:
    :param df:
    :param test_size:
    :return:
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def split_time_series_train_test_val(
    df: pd.DataFrame, target: str, test_size=0.1, val_size=0.1
):
    """
    Split time series data into train, test, val without shuffle
    :param df:
    :param target:
    :param test_size:
    :param val_size:
    :return:
    """
    train_size = 1 - test_size
    val_size = round(val_size / train_size, 2)

    x = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=None, shuffle=False
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=None, shuffle=False
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def split_time_series_train_test(df: pd.DataFrame, target: str, test_size=0.3):
    """
    Split time series data into train and test without shuffle
    :param df:
    :param target:
    :param test_size:
    :return:
    """
    x = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=None, shuffle=False
    )
    return x_train, y_train, x_test, y_test


def split_dataframe_by_date_with_target(df: pd.DataFrame, date_column: str, target_column: str, ratio=0.7):
    """
    Split a DataFrame into training and testing sets based on a datetime column, including target variable.

    :param df: DataFrame to be split
    :param date_column: Name of the datetime column used for splitting
    :param target_column: Name of the target variable column
    :param ratio: Ratio for training and testing split (default is 0.7)
    :return: Tuple containing X_train, X_test, y_train, y_test
    """
    # Sort the DataFrame based on the date_column
    sorted_df = df.sort_values(by=date_column)

    # Calculate the index for splitting
    split_index = int(len(sorted_df) * ratio)

    # Split the DataFrame
    train_df = sorted_df.iloc[:split_index]
    test_df = sorted_df.iloc[split_index:]

    # Split the independent variables and target variable
    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return x_train, x_test, y_train, y_test
