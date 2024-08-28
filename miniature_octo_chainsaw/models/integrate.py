import matplotlib.pyplot as plt
import autograd.numpy as np
from scipy.integrate import solve_ivp
from miniature_octo_chainsaw.logging_ import logger

plt.style.use("ggplot")


def solve_ivp_(model: object, method: str = "RK45", plot: bool = True) -> np.ndarray:
    """
    Numerically integrate the model equations to get a time-series solution.

    Parameters
    ----------
    model : object
        instance of the Model object
    method : str
        integrator method
    plot : bool
        whether to plot the solution

    Returns
    -------
    object : solution
    """

    def model_rhs(_, y):
        return model.rhs_(y)

    y0 = model.initial_state
    t_span = model.integration_interval

    model.mask["compartments"] = True
    logger.debug("Integrate the model equations to get a solution estimate.")
    sol = solve_ivp(model_rhs, y0=y0, t_span=t_span, method=method)

    if plot:
        plot_compartment = model.compartments.index(model.plot_compartment)
        plt.plot(sol.t, sol.y[plot_compartment, :])
        plt.xlabel("time")
        plt.ylabel(model.plot_compartment)
        plt.show()

    logger.debug(f"Model equations were integrated upto time {sol.t[-1]}.")
    logger.debug(f"Steady state estimation from integration: {sol.y[:, -1]}")

    return sol.y[:, -1]
