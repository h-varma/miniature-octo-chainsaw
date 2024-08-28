import autograd.numpy as anp
from parameter_estimation.models.indexing import get_compartment_indices
from parameter_estimation.models.helper import get_parameter_dict
from parameter_estimation.models.lorenz import settings


def model_rhs(y, problem, parameters=None):
    p = get_parameter_dict(y, problem, parameters, settings)
    c_idx = get_compartment_indices(problem, settings)
    x, y, z = y[c_idx]

    model_equations = {
        "x": p['sigma'] * (-x + y),
        "y": p['r'] * x - y - x * z,
        "z": -p['b'] * z + x * y
    }

    return anp.array([model_equations[key] for key in settings.compartments])


def model_jacobian(y, problem, parameters=None):
    p = get_parameter_dict(y, problem, parameters, settings)
    c_idx = get_compartment_indices(problem, settings)
    x, y, z = y[c_idx]

    model_jacobian = {
        'x': anp.array([-p['sigma'],
                        p['sigma'],
                        0]),
        'y': anp.array([p['r'] - z,
                        -1,
                        -x]),
        'z': anp.array([y,
                        x,
                        -p['b']])
    }

    return anp.row_stack([model_jacobian[key] for key in settings.compartments])
