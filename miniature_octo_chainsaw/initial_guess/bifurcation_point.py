import autograd.numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from ..continuation.select_continuer import import_continuer
from ..optimization.single_experiment.select_optimizer import import_optimizer
from ..parameter_estimation.problem_generator import OptimizationProblemGenerator
from ..models.utils import dict_to_nparray, nparray_to_dict
from ..logging_ import logger


def find_bifurcation_point(
    x0: np.ndarray,
    model: dataclass,
    continuer_name: str = "deflated",
    optimizer_name: str = "scipy",
) -> object:
    """
    Find a bifurcation point starting from the steady state x0.

    Parameters
    ----------
    x0 : np.ndarray
        starting point
    model : dataclass
        details of the model
    continuer_name : str
        continuation method
    optimizer_name : str
        name of the local optimizer

    Returns
    -------
    object : results of the continuation
    """

    # continue the steady state solutions along the homotopy parameter
    parameter = model.controls["homotopy"]

    model.mask["controls"] = True
    model.parameters[parameter]["vary"] = True

    Continuer = import_continuer(continuer_name)

    steady_states = Continuer(
        func=model.rhs_,
        x0=x0,
        p0=model.parameters[parameter]["value"],
        p_min=model.continuation_settings["h_min"],
        p_max=model.continuation_settings["h_max"],
        p_step=model.continuation_settings["h_step"],
        p_idx=len(x0),
        local_optimizer=optimizer_name,
    )

    idx = model.compartments.index(model.to_plot)
    for parameter_value, solutions in zip(steady_states.parameters, steady_states.solutions):
        for solution in solutions:
            plt.plot(parameter_value, solution[idx], "ok")
    plt.xlabel(model.controls["homotopy"])
    plt.ylabel(model.to_plot)
    plt.show()

    # detect the bifurcation point from the continuation results
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
    approx = dict_to_nparray(c=c, p=p, h=h, model=model)

    Optimizer = import_optimizer(optimizer_name)

    Objective = OptimizationProblemGenerator(model, include_singularity=True)
    objective_function = Objective.stack_functions

    Constraints = OptimizationProblemGenerator(
        model, include_steady_state=True, include_normalization=True
    )
    equality_constraints = Constraints.stack_functions

    logger.debug(f"Get the bifurcation point near {p[parameter]} using {optimizer_name} optimizer.")
    optimizer = Optimizer(
        objective=objective_function,
        x0=approx,
        constraints={"type": "eq", "fun": equality_constraints},
    )

    optimizer.minimize()

    if optimizer.result.success:
        solution = optimizer.result.x
        max_obj = np.linalg.norm(objective_function(solution), ord=np.inf)
        if not np.isclose(max_obj, 0):
            logger.warning(f"Objective function is satisfied only upto {max_obj:.3e}")
        _, p, _ = nparray_to_dict(x=solution, model=model)
        logger.info(f"Found a bifurcation point at {p[parameter]}.")
        return solution
    else:
        raise RuntimeError("Could not find a bifurcation point!")
