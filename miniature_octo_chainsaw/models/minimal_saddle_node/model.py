import autograd.numpy as anp
from parameter_estimation.models.indexing import get_compartment_indices
from parameter_estimation.models.helper import get_parameter_dict
from parameter_estimation.models.minimal_saddle_node import settings


def model_rhs(y, problem, parameters=None):
    p = get_parameter_dict(y, problem, parameters, settings)
    c_idx = get_compartment_indices(problem, settings)
    x, y = y[c_idx]

    model_equations = {
        'x': 2 * p['k1'] * y - p['k2'] * (x ** 2) - p['k3'] * x * y - p['k4'] * x,
        'y': p['k2'] * (x ** 2) - p['k1'] * y
    }

    return anp.array([model_equations[key] for key in settings.compartments])


def model_jacobian(y, problem, parameters=None):
    p = get_parameter_dict(y, problem, parameters, settings)
    c_idx = get_compartment_indices(problem, settings)
    x, y = y[c_idx]

    model_jacobian = {
        'x': anp.array([-p['k4'] - 2 * p['k2'] * x - p['k3'] * y,
                        2 * p['k1'] - p['k3'] * x]),
        'y': anp.array([2 * p['k2'] * x,
                        -p['k1']])
    }

    return anp.row_stack([model_jacobian[key] for key in settings.compartments])
