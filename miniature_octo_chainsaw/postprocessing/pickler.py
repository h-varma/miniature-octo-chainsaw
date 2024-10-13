import os
import time
import pickle
import easygui
from miniature_octo_chainsaw.parameter_estimation.results import get_results


def create_folder_for_results(path: str):
    """
    Create a folder for storing the results.

    Parameters
    ----------
    path : str
        path to the model folder

    Returns
    -------
    str : path to the newly created folder
    """
    folder_path = path + "\\results"
    if not os.path.exists(folder_path):
        os.mkdir(path + "\\results")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        os.mkdir(folder_path + "\\" + timestamp)
    except FileExistsError:
        raise FileExistsError(f"The folder {timestamp} already exists.")

    return folder_path + "\\" + timestamp + "\\"


def save_results_as_pickle(res: object, path: str):
    """
    Pickle the parameter estimation results.

    Parameters
    ----------
    res : object
        parameter estimation results
    path : str
        path to the folder where the results are to be stored
    """
    with open(path + "summary.pkl", "wb") as f:
        pickle.dump(get_results(res), f)


def load_results_from_pickle():
    """
    Load the parameter estimation results from pickle.

    Returns
    -------
    dict : results
    """
    file_path = easygui.fileopenbox()
    with open(file_path, "rb") as f:
        return pickle.load(f)
