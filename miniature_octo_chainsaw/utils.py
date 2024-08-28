from functools import wraps
from time import time

import numpy as np


def where_zero(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Return indices of x where x is zero.

    Parameters
    ----------
    x : np.ndarray
        input array
    tol : float
        tolerance for zero

    Returns
    -------
    np.ndarray : indices of x where x is almost zero
    """
    return np.where(np.isclose(x, 0, atol=tol))[0]


def where_positive(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Return indices of x where x is positive.

    Parameters
    ----------
    x : np.ndarray
        input array
    tol : float
        tolerance for zero

    Returns
    -------
    np.ndarray : indices of x where x is positive
    """
    return np.where(x > tol)[0]


def where_negative(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Return indices of x where x is negative.
    
    Parameters
    ----------
    x : np.ndarray
        input array
    tol : float
        precision for negative
    """
    return np.where(x < -tol)[0]


def timing_decorator(func: callable) -> callable:
    """
    Decorator that prints execution time of a function.

    Parameters
    ----------
    func : callable
        function to be timed.

    Returns
    -------
    callable : wrapped function.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.3f} seconds.")
        return result

    return wrapped_func
