import autograd.numpy as np
from ..models.utils import nparray_to_dict


def match_solutions_to_data(model: object, solutions: list) -> np.ndarray:
    """
    Match the solutions of two-parameter continuation to the experimental data.

    Parameters
    ----------
    model : object
        details of the model
    solutions : list
        solutions from the two-parameter continuation

    Returns
    -------
    np.ndarray : initial guess
    np.ndarray : boolean array to mask unused data points
    """
    if len(solutions) == 0:
        raise RuntimeError("No solutions found in the two-parameter continuation!")

    h_param = model.controls["homotopy"]
    f_param = model.controls["free"]

    solutions = sorted(solutions, key=lambda x: get_parameter_value(x, type_="f", model=model))
    f_params = list(map(lambda x: get_parameter_value(x, type_="f", model=model), solutions))
    h_params = list(map(lambda x: get_parameter_value(x, type_="h", model=model), solutions))

    h_data = np.array([d[h_param] for d in model.data])
    f_data = np.array([d[f_param] for d in model.data])

    initial_guess = []
    mask = np.ones(len(model.data))

    for h_value in np.unique(h_data):
        where_h = np.where(np.isclose(h_params, h_value))[0]
        count_h = len(where_h)

        if count_h == 2:
            matching_f = f_data[np.isclose(h_data, h_value)]
            if len(matching_f) == 2:
                for idx in where_h:
                    initial_guess.append(solutions[idx])
            elif len(matching_f) == 1:
                f_values = np.array(f_params)[where_h]
                f_closest = min(f_values, key=lambda x: abs(x - matching_f))
                idx = f_params.index(f_closest)
                initial_guess.append(solutions[idx])

        elif count_h == 1:
            where_h = where_h[0]
            if len([h for h in h_data if h == h_value]) == 2:
                _f_data = f_data[np.isclose(h_data, h_value)]
                f = max(_f_data, key=lambda x: abs(x - f_params[where_h]))
                mask[np.isclose(h_data, h_value) & np.isclose(f_data, f)] = 0
            initial_guess.append(solutions[where_h])

        elif count_h == 0:
            mask[np.isclose(h_data, h_value)] = 0

    # sort the initial guesses and experimental data so that they match
    initial_guess.sort(key=lambda x: get_parameter_value(x, type_="h", model=model))
    iterable_ = zip(model.data, mask)
    iterable_ = sorted(iterable_, key=lambda x: x[0][h_param])

    model.data, mask = (list(x) for x in zip(*iterable_))
    return np.hstack(initial_guess), mask


def get_parameter_value(x: np.ndarray, type_: str, model: object) -> float:
    """
    Get the value of homotopy or free parameter from the solution array.

    Parameters
    ----------
    x : np.ndarray
        solution array
    type_ : str
        parameter type
    model : object
        details of the model

    Returns
    -------
    float : parameter value
    """
    if type_ == "h":
        type_ = "homotopy"
    elif type_ == "f":
        type_ = "free"
    else:
        raise ValueError("Unrecognized parameter type!")

    _, p, _ = nparray_to_dict(x, model=model)
    return p[model.controls[type_]]
