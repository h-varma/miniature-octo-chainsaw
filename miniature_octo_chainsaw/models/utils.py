import autograd.numpy as np
from typing import Union, Tuple, List


def nparray_to_dict(x: np.ndarray, model: object) -> Tuple[dict, dict, dict]:
    """
    Sorts the content of a numpy array into dictionaries.

    Parameters
    ----------
    x : numpy array
        model state
    model : object
        model information, parameters and settings

    Returns
    -------
    c : dict
        compartments
    p : dict
        parameters
    h : dict
        auxiliary variables
    """

    c = dict()
    len_ = 0
    if model.mask["compartments"]:
        len_ += len(model.compartments)
        c_idx = slice(0, len(model.compartments))
        for i, x_ in enumerate(x[c_idx]):
            c[model.compartments[i]] = x_
    idx = len(model.compartments) if model.mask["compartments"] else 0

    p = {key: item["value"] for key, item in model.parameters.items()}
    if model.mask["controls"]:
        for key in model.controls.values():
            if model.parameters[key]["vary"]:
                len_ += 1
                p[key] = x[idx]
                idx += 1

    h = dict()
    if model.mask["auxiliary_variables"]:
        h = assign_auxiliary_variables(x, model=model, idx=idx)
        if model.bifurcation_type == "saddle-node":
            len_ += len(model.compartments)
        elif model.bifurcation_type == "hopf":
            len_ += 2 * len(model.compartments) + 1

    if model.mask["global_parameters"]:
        idx = 0
        for i, key in enumerate(model.global_parameters):
            if model.parameters[key]["vary"]:
                p[key] = x[-len(model.global_parameters) + idx]
                idx += 1
                len_ += 1

    assert len_ == len(x)

    return c, p, h


def dict_to_nparray(
    c: Union[dict, List[dict]],
    p: Union[dict, List[dict]],
    h: Union[dict, List[dict]],
    model: object,
) -> np.ndarray:
    """
    Sorts the content of dict into a numpy array.

    Parameters
    ----------
    c : dict or list of dicts
        model states
    p : dict or list of dicts
        parameter values
    h : dict or list of dicts
        auxiliary variable values
    model : object
        details of the model

    Returns
    -------

    """

    if isinstance(c, dict) and isinstance(p, dict) and isinstance(h, dict):
        n_experiments = 1
        c, p, h = [c], [p], [h]
    elif isinstance(c, list) and isinstance(p, list) and isinstance(h, list):
        assert len(c) == len(p) == len(h), "The lists must have the same length."
        n_experiments = len(c)
    else:
        n_experiments = 0

    x = np.array([])
    for i in range(n_experiments):
        if model.mask["compartments"]:
            for compartment in model.compartments:
                x = np.hstack((x, c[i][compartment]))

        if model.mask["controls"]:
            for control in model.controls.values():
                if model.parameters[control]["vary"]:
                    x = np.hstack((x, p[i][control]))

        if model.mask["auxiliary_variables"]:
            h = assign_auxiliary_variables(h[i], model=model, idx=0)
            x = np.hstack((x, h))

    if model.mask["global_parameters"]:
        for global_parameter in model.global_parameters:
            if model.parameters[global_parameter]["vary"]:
                x = np.hstack((x, p[global_parameter]))

    return x


def assign_auxiliary_variables(
    x: Union[np.ndarray, dict], model: object, idx: int = 0
) -> Union[np.ndarray, dict]:
    if isinstance(x, dict):
        y = np.array([])
        if model.bifurcation_type == "hopf":
            for key in ["v", "w", "mu"]:
                y = np.hstack((y, x[key]))
        elif model.bifurcation_type == "saddle-node":
            y = np.hstack((y, x["h"]))
        return y
    else:
        y = dict()
        if model.bifurcation_type == "hopf":
            y["v"] = x[slice(idx, idx + len(model.compartments))]
            idx += len(model.compartments)
            y["w"] = x[slice(idx, idx + len(model.compartments))]
            idx += len(model.compartments)
            y["mu"] = x[slice(idx, idx + 1)]
            idx += 1
        elif model.bifurcation_type == "saddle-node":
            y["h"] = x[slice(idx, idx + len(model.compartments))]
        return y
