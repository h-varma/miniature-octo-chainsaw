from itertools import chain
import numpy as np
import pandas as pd
from miniature_octo_chainsaw.logging_ import logger

rng = np.random.default_rng(0)


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame, noise: float = 0, length: int = None):
        """
        Preprocess the measurement data.

        Parameters
        ----------
        data : pd.DataFrame
            measurement data
        noise : float
            noise level
        length : int
            number of data points
        """
        self.data = data
        self.noise = noise
        self.length = length

        self.__select_subset()
        if noise > 0:
            self.__add_noise()

    def __add_noise(self):
        """Add noise to the synthetic measurement data. """
        if not isinstance(self.data, pd.DataFrame):
            raise Exception("Data must be a pandas DataFrame.")
        control_1, control_2 = self.data.columns
        noisy_data = [[], []]
        for d1, d2 in zip(self.data[control_1], self.data[control_2]):
            d1 = rng.normal(d1, self.noise * d1)
            if type(d2) is str:
                for i, c2_ in enumerate(d2.split(",")):
                    c2_ = rng.normal(float(c2_), self.noise * float(c2_))
                    noisy_data[i].append({control_1: d1, control_2: c2_})
            else:
                d2 = rng.normal(d2, self.noise * d2)
                noisy_data[0].append({control_1: d1, control_2: d2})
        self.data = list(chain(*noisy_data))


    def __select_subset(self):
        """Filter out a subset of the measurement data. """
        if self.length > len(self.data):
            logger.warning(
                f"{self.length} data points not available. "
                f"Using all available data points."
            )

        elif self.length is not None:
            filter_idx = np.random.choice(len(self.data), size=self.length, replace=False)
            filter_idx = np.sort(filter_idx)

            if type(self.data) is pd.DataFrame:
                self.data = self.data.iloc[filter_idx]
            elif type(self.data) is list:
                self.data = [d for i, d in enumerate(self.data) if i in filter_idx]
            else:
                raise Exception("Data must be a list or pandas DataFrame.")
