import autograd.numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from miniature_octo_chainsaw.continuation.select_continuer import import_continuer
from miniature_octo_chainsaw.parameter_estimation.problem_generator import OptimizationProblemGenerator
from miniature_octo_chainsaw.models.utils import dict_to_nparray, nparray_to_dict


def trace_measured_bifurcations(
    x0: np.ndarray,
    model: object,
    continuer_name: str = "pseudo-arclength",
):
    """
    Continue the bifurcation point to draw a two-parameter bifurcation diagram.

    Parameters
    ----------
    x0 : np.ndarray
        starting bifurcation point
    model : object
        model details
    continuer_name : str
        continuation method

    Returns
    -------
    np.ndarray : set of bifurcation points in two-parameters
    """
    Continuer = import_continuer(continuer_name)
    homotopy_parameter = model.controls["homotopy"]
    free_parameter = model.controls["free"]

    data = np.unique([d[homotopy_parameter] for d in model.data])

    c, p, h = nparray_to_dict(x=x0, model=model)
    p0 = p[homotopy_parameter]
    p_idx = len(model.compartments)

    model.parameters[free_parameter]["vary"] = True

    ContinuationProblem = OptimizationProblemGenerator(
        model=model,
        include_steady_state=True,
        include_singularity=True,
        include_normalization=True,
    )
    objective_function = ContinuationProblem.stack_functions

    x0 = dict_to_nparray(c=c, p=p, h=h, model=model)
    x0 = np.delete(x0, obj=p_idx)

    continuer = Continuer(
        func=objective_function,
        x0=x0,
        p0=p0,
        p_min=model.continuation_settings["h_min"],
        p_max=2 * max(np.abs(data)),
        p_step=model.continuation_settings["h_step"],
        p_step_min=(max(data) - min(data)) * 1e-8,
        p_step_max=max(np.abs(np.diff(data))),
        p_idx=p_idx,
        fast_iters=3,
        data=data,
    )

    solutions = []
    for parameter, solution in zip(continuer.parameters, continuer.solutions):
        model.parameters[homotopy_parameter]["vary"] = False
        c, p, h = nparray_to_dict(x=solution, model=model)
        p[homotopy_parameter] = parameter
        plt.plot(p[homotopy_parameter], p[free_parameter], "ok")

        model.parameters[homotopy_parameter]["vary"] = True
        solution = dict_to_nparray(c=c, p=p, h=h, model=model)
        solutions.append(solution)
    plt.xlabel(homotopy_parameter)
    plt.ylabel(free_parameter)
    plt.show()

    return solutions
