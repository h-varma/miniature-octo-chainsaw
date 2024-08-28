from scipy.optimize import minimize_scalar
import autograd.numpy as np


def line_search(x: np.ndarray, dx: np.ndarray, func: callable, strategy: str) -> float:
    if strategy == "exact":

        def step_finder(t_):
            return func(x + t_ * dx)

        result = minimize_scalar(step_finder, method="Bounded", bounds=(0, 1))
        assert result.success, "Exact line search did not converge."
        return result.x

    else:
        raise ValueError(f"{strategy} line search strategy is not supported.")
