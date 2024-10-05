from functools import wraps
from time import time


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
