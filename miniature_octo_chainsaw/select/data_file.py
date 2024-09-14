import pandas as pd


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
    file_path = ".\\models\\"
    if name == "predator_prey":
        file_path += name + "\\Ni_vs_delta.dat"
        return pd.read_table(file_path, sep=" ")

    elif name == "peroxidase_oxidase":
        file_path += name + "\\k4_vs_k5.dat"
        return pd.read_table(file_path, sep=" ")

    elif name == "steinmetz_larter":
        file_path += name + "\\k7_vs_k8.dat"
        return pd.read_table(file_path, sep=" ")

    else:
        raise Exception(f"No data file found for the {name} model.")
