import numpy as np
import pandas as pd
from itertools import chain
from ..logging_ import logger

rng = np.random.default_rng(0)


class DataPreprocessor:
    def __init__(self):
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
        self._data = None
        self._noise = None
        self._length = None

    def load_the_data(self, file_path: str):
        """
        Load the measurement data. 
        
        Parameters
        ----------
        file_path : str
            path to the data file
        """
        if self._data is not None:
            logger.warning("Data already loaded. Overwriting the data.")

        self._data = pd.read_table(file_path + "/data.dat", sep=" ")
        logger.info(f"Loaded data from {file_path}.")

    def __add_noise(self):
        """Add noise to the synthetic measurement data. """
        if not isinstance(self._data, pd.DataFrame):
            raise Exception("Data must be a pandas DataFrame.")
        control_1, control_2 = self._data.columns
        noisy_data = [[], []]
        for d1, d2 in zip(self._data[control_1], self._data[control_2]):
            d1 = rng.normal(d1, self._noise * d1)
            if type(d2) is str:
                for i, c2_ in enumerate(d2.split(",")):
                    c2_ = rng.normal(float(c2_), self._noise * float(c2_))
                    noisy_data[i].append({control_1: d1, control_2: c2_})
            else:
                d2 = rng.normal(d2, self._noise * d2)
                noisy_data[0].append({control_1: d1, control_2: d2})
        self._data = list(chain(*noisy_data))
        logger.info(f"Loaded {len(self._data)} data points with {self._noise * 100}% noise.")

    def add_noise_to_the_data(self, scale: float):
        """Add noise to the measurement data. """
        self._noise = scale
        self.__add_noise()

    def __select_subset(self):
        """Filter out a subset of the measurement data. """
        if self._length > len(self._data):
            logger.warning(
                f"{self._length} data points not available. "
                f"Using all available data points."
            )

        elif self._length is not None:
            filter_idx = np.random.choice(len(self._data), size=self._length, replace=False)
            filter_idx = np.sort(filter_idx)

            if type(self._data) is pd.DataFrame:
                self._data = self._data.iloc[filter_idx]
            elif type(self._data) is list:
                self._data = [d for i, d in enumerate(self._data) if i in filter_idx]
            else:
                raise Exception("Data must be a list or pandas DataFrame.")
            
        logger.info(f"Selected {len(self._data)} data points.")

    def select_subset_of_data(self, length: int):
        """Filter out a subset of the measurement data. """
        self._length = length
        self.__select_subset()

    @property
    def data(self):
        return self._data
