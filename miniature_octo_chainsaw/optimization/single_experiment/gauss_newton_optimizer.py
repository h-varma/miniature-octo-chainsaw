import autograd.numpy as np
import inspect
import warnings
from typing import Union
from ...optimization.single_experiment.gauss_newton_method import GeneralizedGaussNewton
from ...optimization.single_experiment.base_optimizer import BaseOptimizer


class GaussNewtonOptimizer(BaseOptimizer):
    """
    Use the in-house Gauss-Newton implementation to solve the optimization problem.
    """

    def __init__(
        self,
        objective: callable,
        x0: np.ndarray,
        lb: Union[np.ndarray, list[float]] = None,
        ub: Union[np.ndarray, list[float]] = None,
        constraints: dict = None,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        objective : callable
            objective function to minimize
        x0: np.ndarray
            initial guess
        lb: Union[np.ndarray, list[float]]
            lower bounds
        ub: Union[np.ndarray, list[float]]
            upper bounds
        constraints: dict
            constraints
        """
        super().__init__()
        self.x0 = x0
        self.objective = objective
        self.constraints = constraints
        if lb is not None and ub is not None and len(lb) != len(ub):
            raise ValueError("Lower bounds must be the same length as upper bounds.")
        self.lb, self.ub = lb, ub

        self.options = {}

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
        self._set_options(options)
        if self.lb is not None or self.ub is not None:
            if any(np.isfinite(self.lb)) or any(np.isfinite(self.ub)):
                self._bounds_to_constraints()

        def equality_constraints(x):
            return self._join_constraints_by_type(x, type_="eq")

        def inequality_constraints(x):
            return self._join_constraints_by_type(x, type_="ineq")

        constraints = {
            "equality": equality_constraints,
            "inequality": inequality_constraints,
        }

        optimizer = GeneralizedGaussNewton(
            objective=self.objective,
            constraints=constraints,
            x0=self.x0,
            **self.options,
        )
        self.result = optimizer.result

    def _bounds_to_constraints(self):
        """
        Reformulate bounds as inequality constraints.
        """

        def bounds(y):
            constr = np.array([])
            for i, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                if np.isfinite(lb):
                    constr = np.concatenate((constr, np.array([y[i] - lb])))
                if np.isfinite(ub):
                    constr = np.concatenate((constr, np.array([ub - y[i]])))
            return constr

        constraint = {"type": "ineq", "fun": bounds, "jac": None, "args": None}

        if self.constraints is None:
            self.constraints = []
        elif isinstance(self.constraints, dict):
            self.constraints = [self.constraints]
        self.constraints.append(constraint)

    def _join_constraints_by_type(self, x: np.ndarray, type_: str) -> np.ndarray:
        """
        Combines all constraints of one type into an array.

        Parameters
        ----------
        x : numpy array
            constraint evaluation value
        type_ : str
            constraint type ("eq" or "ineq")

        Returns
        -------
        numpy array : values of the constraints
        """

        if self.constraints is None:
            return None
        elif isinstance(self.constraints, dict):
            self.constraints = [self.constraints]
        constr = np.array([])
        for _constr in self.constraints:
            if _constr["type"] == type_:
                constr = np.concatenate((constr, _constr["fun"](x)))
        if len(constr) == 0:
            return None
        return constr

    def _set_options(self, options: dict = None):
        """
        Set the optimizer options.

        Parameters
        ----------
        options: dict
            solver options
        """
        if options is None:
            options = dict()
        specs = inspect.signature(GeneralizedGaussNewton.__init__)
        for param in specs.parameters.values():
            in_options = bool(param.name in options.keys())
            default_empty = bool(param.default is param.empty)
            if in_options:
                self.options[param.name] = options[param.name]
            if not in_options and not default_empty:
                self.options[param.name] = param.default

        invalid = [key for key in options.keys() if key not in self.options.keys()]
        if len(invalid) > 0:
            warnings.warn("Options " + ", ".join(invalid) + " are undefined.")
