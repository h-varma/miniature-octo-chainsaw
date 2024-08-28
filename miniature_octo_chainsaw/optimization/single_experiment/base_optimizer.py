import autograd.numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass


class BaseOptimizer(ABC):
    """
    Optimizer interface - not functional on its own.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize the optimizer.
        """
        self.result = OptimizerResult()
        self.result.level_functions = []

    @abstractmethod
    def minimize(self, method: str = None, options: dict = None):
        """
        Minimize the objective function subject to bounds and constraints.

        Parameters
        ----------
        method : str
            for consistency, non-functional for GaussNewton
        options : dict
            solver options
        """
        raise NotImplementedError


@dataclass
class OptimizerResult:
    """
    Optimizer result object.

    Attributes
    ----------
    x : np.ndarray
        solution of the optimizer
    success : bool
        whether the optimizer has converged
    message : str
        cause of termination
    fun : np.ndarray
        objective (or level) function at solution
    jac : np.ndarray
        value of the jacobian at the solution
    hess : np.ndarray
        value of the hessian at the solution
    hess_inv : np.ndarray
        value of the hessian inverse at the solution
    n_iters : int
        number of iterations performed
    max_cv : float
        maximum constraint violation
    """

    x = None
    success = False
    message = "Optimization has not been attempted."
    fun = None
    jac = None
    hess = None
    hess_inv = None
    n_iters = 0
    max_cv = np.inf
    level_functions = None
