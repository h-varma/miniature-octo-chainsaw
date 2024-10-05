import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from miniature_octo_chainsaw.optimization.single_experiment.select_optimizer import import_optimizer
from miniature_octo_chainsaw.logging_ import logger


def find_steady_state(
        model: object,
        optimizer_name: str = "gauss-newton",
        optimizer_options: dict = None,
        ) -> np.ndarray:
    """
    Integrate and solve the model equations to get a steady state solution.

    Parameters
    ----------
    model : object
        instance of the Model object
    optimizer_method : str
        name of the local optimizer
    optimizer_options : dict
        optimizer-specific options

    Returns
    -------
    np.ndarray : steady state solution
    """

    # integrate the model equations to get a solution estimate
    y0 = model.initial_state
    t_span = model.integration_interval

    model.mask["compartments"] = True
    logger.debug("Integrate the model equations to get a solution estimate.")
    sol = solve_ivp(lambda _, y: model.rhs_(y), y0=y0, t_span=t_span)

    compartment_idx = model.compartments.index(model.plot_compartment)
    plt.plot(sol.t, sol.y[compartment_idx, :])
    plt.xlabel("time")
    plt.ylabel(model.plot_compartment)
    plt.show()

    logger.debug(f"Model equations were integrated upto time {sol.t[-1]}.")
    logger.debug(f"Steady state estimation from integration: {sol.y[:, -1]}")

    # solve the model equations to get the steady state
    x0 = sol.y[:, -1]
    lb = np.zeros_like(x0) if model.non_negative else np.ones_like(x0) * -np.inf
    ub = np.ones_like(x0) * np.inf

    logger.debug(f"Solve model equations using {optimizer_name} to get steady state.")
    Optimizer = import_optimizer(optimizer_name)
    optimizer = Optimizer(model.rhs_, x0=x0, lb=lb, ub=ub)
    optimizer.minimize(options=optimizer_options)

    if optimizer.result.success:
        solution = optimizer.result.x
        max_rhs = np.linalg.norm(model.rhs_(solution), ord=np.inf)
        if not np.isclose(max_rhs, 0):
            warnings.warn(f"RHS is satisfied only upto {max_rhs:.3e}", RuntimeWarning)

        solution_dict = {c: solution[i] for i, c in enumerate(model.compartments)}
        logger.debug(f"Steady state found: {solution_dict}")

        return solution

    raise RuntimeError("Could not solve model equations to find steady state!")
