from itertools import chain
from typing import Union
import numpy as np
import pandas as pd
from miniature_octo_chainsaw.logging_ import logger

rng = np.random.default_rng(0)


def add_noise_to_data(data: pd.DataFrame, noise_scale: float) -> list:
    """
    Add noise to the synthetic measurement data.

    Args:
        data: synthetic "measurement" data
        noise_scale: percentage of noise added to the data

    Returns:
        list of dictionaries containing the noisy data
    """
    control_1, control_2 = data.columns
    noisy_data = [[], []]
    for c1, c2 in zip(data[control_1], data[control_2]):
        c1 = rng.normal(c1, noise_scale * c1)
        if type(c2) is str:
            for i, c2_ in enumerate(c2.split(",")):
                c2_ = rng.normal(float(c2_), noise_scale * float(c2_))
                noisy_data[i].append({control_1: c1, control_2: c2_})
        else:
            c2 = rng.normal(c2, noise_scale * c2)
            noisy_data[0].append({control_1: c1, control_2: c2})
    return list(chain(*noisy_data))


def filter_data(
    data: Union[list, pd.DataFrame], desired_length: int
) -> Union[list, pd.DataFrame]:
    """
    Filter out a subset of the measurement data.

    Args:
        data: synthetic "measurement" data
        desired_length: desired size of the data subset

    Returns:
        subset of the measurement data
    """
    if desired_length > len(data):
        logger.warn(
            f"{desired_length} data points not available. "
            f"Using all available data points."
        )
        return data
    filter_idx = np.random.choice(len(data), size=desired_length, replace=False)
    filter_idx = np.sort(filter_idx)
    if type(data) is pd.DataFrame:
        return data.iloc[filter_idx]
    elif type(data) is list:
        return [d for i, d in enumerate(data) if i in filter_idx]
    else:
        raise Exception("Measurement data type not supported.")
