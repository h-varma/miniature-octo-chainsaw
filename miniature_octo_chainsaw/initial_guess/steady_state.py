import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from ..logging_ import logger


def find_steady_state(model: object) -> np.ndarray:
    """
    Integrate and solve the model equations to get a steady state solution.

    Parameters
    ----------
    model : object
        instance of the Model object

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

    compartment_idx = model.compartments.index(model.to_plot)
    plt.plot(sol.t, sol.y[compartment_idx, :])
    plt.xlabel("time")
    plt.ylabel(model.to_plot)
    plt.show()

    logger.debug(f"Model equations were integrated upto time {sol.t[-1]}.")
    logger.debug(f"Steady state estimation from integration: {sol.y[:, -1]}")

    # solve the model equations to get the steady state
    x0 = sol.y[:, -1]
    logger.debug(f"Solve model equations using scipy.optimize.root to get steady state.")
    res = root(model.rhs_, x0=x0)

    if res.success:
        solution = res.x
        solution_dict = {c: solution[i] for i, c in enumerate(model.compartments)}
        logger.debug(f"Steady state found: {solution_dict}")

        return solution

    raise RuntimeError("Could not solve model equations to find steady state!")
