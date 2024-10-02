from scipy.optimize import minimize_scalar
import autograd.numpy as np
from autograd import grad


def line_search(x: np.ndarray, dx: np.ndarray, func: callable, strategy: str) -> float:
    """
    Find the step length using the line search.

    Parameters
    ----------
    x : np.ndarray
        current point
    dx : np.ndarray
        search direction
    func : callable
        objective function
    strategy : str
        line search strategy

    Returns
    -------
    float : step length
    """

    if strategy == "exact":

        def step_finder(t_):
            return func(x + t_ * dx)

        result = minimize_scalar(step_finder, method="Bounded", bounds=(0, 1))
        assert result.success, "Exact line search did not converge."
        return result.x

    elif strategy == "armijo-backtracking":
        f = func(x)
        df = grad(func)(x)
        t = 1
        beta = 0.8
        gamma = 0.1
        while func(x + t * dx) >= f + gamma * t * (df.T @ dx):
            t *= beta
        return t

    else:
        raise ValueError(f"{strategy} line search strategy is not supported.")
