import os

import pandas as pd
from pandas import read_table


def import_model(name: str) -> object:
    """
    Imports the model equations, jacobian and settings.

    Parameters
    ----------
    name : str
        name of the model

    Returns
    -------
    Model : class
        model class with rhs and jacobian
    """
    if name == "predator_prey":
        from miniature_octo_chainsaw.models.predator_prey.model import Model
    elif name == "peroxidase_oxidase":
        from miniature_octo_chainsaw.models.peroxidase_oxidase.model import Model
    elif name == "steinmetz_larter":
        from miniature_octo_chainsaw.models.steinmetz_larter.model import Model
    else:
        raise Exception(f"{name} model not found.")
    model = Model()
    return model


def import_data(name: str) -> pd.DataFrame:
    """
    Imports the data from .dat files.
    Parameters
    ----------
    name : str
        name of the model

    Returns
    -------
    data : pd.DataFrame
        data frame with the measurement data
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    if name == "predator_prey":
        file_path += "\\" + name + "\\Ni_vs_delta.dat"
        return read_table(file_path, sep=" ")
    elif name == "peroxidase_oxidase":
        file_path += "\\" + name + "\\k4_vs_k5.dat"
        return read_table(file_path, sep=" ")
    elif name == "steinmetz_larter":
        file_path += "\\" + name + "\\k7_vs_k8.dat"
        return read_table(file_path, sep=" ")
    else:
        raise Exception(f"No data file found for the {name} model.")
