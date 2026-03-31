# This module defines functions to plot data from a dataframe
from abc import ABC


class Plot(ABC):
    """
    Class to plot data. This is an abstract class
    """
    def __init__(self, palette: str = 'Paired'):
        """
        Initializes an instance of the class.
        :param palette: The name of the color palette to use. Defaults to 'Paired'.
        """
        self.palette = palette

    @staticmethod
    def find_num_rows_cols(feature_length: int) -> tuple[int, int]:
        """
        Find the number of rows and columns for subplots
        :param feature_length: number of features
        :return: number of rows, number of columns
        """
        if feature_length == 7:
            num_cols = 3
            num_rows = 3
            return num_rows, num_cols

        num_cols = 5
        while feature_length % num_cols != 0:
            num_cols -= 1

        num_rows = feature_length // num_cols
        return num_rows, num_cols
