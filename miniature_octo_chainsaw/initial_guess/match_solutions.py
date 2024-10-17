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
    data = []

    unique_h_data, unique_idx = np.unique(h_data, return_index=True)
    for h_value in unique_h_data[unique_idx]:
        where_h = np.where(np.isclose(h_params, h_value))[0]
        count_h = len(where_h)

        if count_h == 2:
            # predicted two bifurcation points with the same homotopy parameter value
            matching_f = f_data[np.isclose(h_data, h_value)]
            f_values = np.array(f_params)[where_h]
            if len(matching_f) == 2:
                # experimental data has two bifurcation points with different free parameter values
                for i in range(2):
                    f_closest = min(f_values, key=lambda x: abs(x - matching_f[i]))
                    idx = f_params.index(f_closest)
                    initial_guess.append(solutions[idx])
                    data.append({h_param: h_value, f_param: matching_f[i]})
            elif len(matching_f) == 1:
                # experimental data has only one corresponding bifurcation point
                f_closest = min(f_values, key=lambda x: abs(x - matching_f))
                idx = f_params.index(f_closest)
                initial_guess.append(solutions[idx])
                data.append({h_param: h_value, f_param: matching_f[0]})

        elif count_h == 1:
            # predicted one bifurcation point with the homotopy parameter value
            where_h = where_h[0]
            _f_data = f_data[np.isclose(h_data, h_value)]
            if len(_f_data) == 2:
                # but experimental data has two corresponding bifurcation points
                f_value = min(_f_data, key=lambda x: abs(x - f_params[where_h]))
            else:
                # and experimental data has one corresponding bifurcation point
                f_value = _f_data[0]
            initial_guess.append(solutions[where_h])
            data.append({h_param: h_value, f_param: f_value})

    assert len(initial_guess) > 0, "No matching solutions found for the experimental data!"
    assert len(initial_guess) == len(data)
    h_params = list(map(lambda x: get_parameter_value(x, type_="h", model=model), initial_guess))
    h_data = [d[h_param] for d in data]
    assert all([np.isclose(hp, hd) for hp, hd in zip(h_params, h_data)])

    model.data = data
    return np.hstack(initial_guess)


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
