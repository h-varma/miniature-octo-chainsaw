import autograd.numpy as np
from miniature_octo_chainsaw.select.single_experiment_optimizer import import_single_experiment_optimizer
from miniature_octo_chainsaw.models.utils import nparray_to_dict, dict_to_nparray
from miniature_octo_chainsaw.parameter_estimation.subproblems import Problem
from miniature_octo_chainsaw.logging_ import logger


def detect_bifurcation_point(steady_states: object, model: object) -> np.ndarray:
    """
    Detect the bifurcation point from the results of the continuation method.

    Parameters
    ----------
    steady_states : object
        results of the one-parameter continuation
    model : object
        details of the model

    Returns
    -------
    np.ndarray : steady states, parameter values, and auxiliary variables
    """

    parameter = model.controls["homotopy"]

    if model.bifurcation_type == "hopf":
        approx = steady_states.detect_hopf_bifurcation(parameter=parameter)
    elif model.bifurcation_type == "saddle-node":
        approx = steady_states.detect_saddle_node_bifurcation(parameter=parameter)
    else:
        raise ValueError("Unrecognized bifurcation type!")
    c, p, h = nparray_to_dict(x=approx, model=model)

    jacobian_ = model.jacobian_(approx)

    eig_vals, eig_vecs = np.linalg.eig(jacobian_)
    mask = (eig_vals.real == min(eig_vals.real, key=abs)) & (eig_vals.imag >= 0)

    if model.bifurcation_type == "hopf":
        h["mu"] = eig_vals.imag[mask]
        h["v"] = eig_vecs.real[:, mask].squeeze()
        h["w"] = eig_vecs.imag[:, mask].squeeze()

    elif model.bifurcation_type == "saddle-node":
        h["h"] = eig_vecs.real[:, mask].squeeze()

    else:
        raise ValueError("Unrecognized bifurcation type!")

    model.mask["auxiliary_variables"] = True
    return dict_to_nparray(c=c, p=p, h=h, model=model)


def solve_bifurcation_condition(
    x0: np.ndarray, model: object, local_optimizer: str = "scipy"
) -> np.ndarray:
    """
    Get a bifurcation point from the results of the continuation method.

    Parameters
    ----------
    x0 : np.ndarray
        starting guess
    model : object
        details of the model
    local_optimizer : str
        name of the local optimizer

    Returns
    -------
    np.ndarray : bifurcation point
    """
    parameter = model.controls["homotopy"]
    c, p, h = nparray_to_dict(x=x0, model=model)

    if model.non_negative:
        lb = dict_to_nparray(
            c={key: 0 for key in c.keys()},
            p={key: 0 for key in p.keys()},
            h={key: -np.inf * np.ones_like(values) for key, values in h.items()},
            model=model,
        )
        ub = np.ones_like(x0) * np.inf
    else:
        lb, ub = None, None

    Optimizer = import_single_experiment_optimizer(local_optimizer)

    Objective = Problem(model, include_singularity=True)
    objective_function = Objective.stack_functions

    Constraints = Problem(model, include_steady_state=True, include_normalization=True)
    equality_constraints = Constraints.stack_functions

    logger.debug(
        f"Get the bifurcation point near {p[parameter]} using {local_optimizer} optimizer."
    )
    optimizer = Optimizer(
        objective=objective_function,
        x0=x0,
        lb=lb,
        ub=ub,
        constraints={"type": "eq", "fun": equality_constraints},
    )

    optimizer.minimize()

    if optimizer.result.success:
        solution = optimizer.result.x
        max_obj = np.linalg.norm(objective_function(solution), ord=np.inf)
        if not np.isclose(max_obj, 0):
            logger.warn(f"Objective function is satisfied only upto {max_obj:.3e}")
        _, p, _ = nparray_to_dict(x=solution, model=model)
        logger.info(f"Found a bifurcation point at {p[parameter]}.")
        return solution
    else:
        raise RuntimeError("Could not find a bifurcation point!")
