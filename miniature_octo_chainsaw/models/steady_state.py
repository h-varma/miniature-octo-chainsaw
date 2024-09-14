import numpy as np
from miniature_octo_chainsaw.select.single_experiment_optimizer import import_single_experiment_optimizer
from miniature_octo_chainsaw.logging_ import logger
import warnings


def solve_rhs(
    model: object,
    x0: np.ndarray,
    local_optimizer: str = "gauss-newton",
    scipy_method: str = None,
    options: dict = None,
):
    """
    Solve the model equations to get a steady state solution.

    Parameters
    ----------
    model : object
        instance of the Model object
    x0 : np.ndarray
        starting guess
    local_optimizer : str
        name of the local optimizer
    scipy_method : str
        name of the scipy optimizer method
    options : dict
        optimizer-specific options

    Returns
    -------
    np.ndarray : steady state solution
    """

    lb = np.zeros_like(x0) if model.non_negative else np.ones_like(x0) * -np.inf
    ub = np.ones_like(x0) * np.inf

    logger.debug(f"Solve model equations using {local_optimizer} to get steady state.")
    Optimizer = import_single_experiment_optimizer(local_optimizer)
    optimizer = Optimizer(model.rhs_, x0=x0, lb=lb, ub=ub)
    optimizer.minimize(method=scipy_method, options=options)

    if optimizer.result.success:
        solution = optimizer.result.x
        max_rhs = np.linalg.norm(model.rhs_(solution), ord=np.inf)
        if not np.isclose(max_rhs, 0):
            warnings.warn(f"RHS is satisfied only upto {max_rhs:.3e}", RuntimeWarning)

        solution_dict = {c: solution[i] for i, c in enumerate(model.compartments)}
        logger.debug(f"Steady state found: {solution_dict}")

        return solution

    raise RuntimeError("Could not solve model equations to find steady state!")
