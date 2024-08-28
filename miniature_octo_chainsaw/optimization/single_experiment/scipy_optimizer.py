import inspect
import autograd.numpy as np
import scipy.optimize
import warnings
from typing import Union, List
from miniature_octo_chainsaw.optimization.single_experiment.base_optimizer import BaseOptimizer


class ScipyOptimizer(BaseOptimizer):
    """
    Use one of the SciPy optimizers to solve the optimization problem.

    Find details on the optimizer and configuration options at: :func:`scipy.optimize.minimize`.
    """

    def __init__(
        self,
        objective: callable,
        x0: np.ndarray,
        lb: Union[np.ndarray, List[float]] = None,
        ub: Union[np.ndarray, List[float]] = None,
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

        if lb is None and ub is None:
            lb = -np.inf * np.ones_like(x0)
            ub = np.inf * np.ones_like(x0)
        self.bounds = scipy.optimize.Bounds(lb, ub)

        self.options = {}

        self.is_least_squares = bool(not isinstance(self.objective(x0), float))

    def minimize(self, method: str = None, options: dict = None):
        """
        Minimize the objective function subject to bounds and constraints.

        Parameters
        ----------
        method : str
            scipy optimization method
        options : dict
            solver options
        """
        self._check_if_valid_method(method=method)

        if self.constraints is None and self.is_least_squares:
            if method is None:
                lower_bounded = any(np.isfinite(self.bounds.lb))
                upper_bounded = any(np.isfinite(self.bounds.ub))
                if not lower_bounded and not upper_bounded:
                    method = "lm"
                else:
                    method = "trf"
            self._set_options(scipy.optimize.least_squares, options=options)

            res = scipy.optimize.least_squares(
                fun=self.objective,
                x0=self.x0,
                method=method,
                bounds=self.bounds,
                **self.options,
            )
        else:
            if (
                self.constraints is not None
                or bool(not isinstance(self.objective(self.x0), float))
                != self.is_least_squares
            ):
                objective = lambda x: np.linalg.norm(self.objective(x)) ** 2
            else:
                objective = self.objective
            self._set_options(scipy.optimize.minimize, options=options)

            res = scipy.optimize.minimize(
                fun=objective,
                x0=self.x0,
                method=method,
                bounds=self.bounds,
                constraints=self.constraints,
                **self.options,
            )

        self._get_results(result=res)

    def _set_options(self, func: callable, options: dict = None):
        """
        Set the optimizer options.

        Parameters
        ----------
        func: callable
            solver function
        options: dict
            solver options
        """
        if options is None:
            options = dict()
        specs = inspect.signature(func)
        for param in specs.parameters.values():
            in_options = bool(param.name in options.keys())
            default_empty = bool(param.default is param.empty)
            in_given = bool(param.name in ["method", "bounds", "constraints"])
            if in_options:
                self.options[param.name] = options[param.name]
            if not in_options and not default_empty and not in_given:
                self.options[param.name] = param.default

        invalid = [key for key in options.keys() if key not in self.options.keys()]
        if len(invalid) > 0:
            warnings.warn("Options " + ", ".join(invalid) + " are undefined.")

    def _check_if_valid_method(self, method: str):
        """
        Check if the given method is available.

        Parameters
        ----------
        method :  str
            optimization method
        """
        if method is None:
            return None

        ls_methods = ["trf", "dogbox", "lm"]
        min_methods = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]

        if self.constraints is None and self.is_least_squares:
            if method.lower() not in [m.lower() for m in ls_methods]:
                if method.lower() in [m.lower() for m in min_methods]:
                    warnings.warn(
                        f"{method} is not a valid least squares method. "
                        f"Re-defining the objective to use scipy.optimize.minimize instead."
                    )
                    self.is_least_squares = False
                else:
                    raise Exception(
                        "Not a valid least squares method. Please choose one of the following methods: "
                        + ", ".join(ls_methods)
                    )
        else:
            if method.lower() not in [m.lower() for m in min_methods]:
                raise Exception(
                    "Not a valid scipy minimize method. Please choose one of the following methods: "
                    + ", ".join(min_methods)
                )

    def _get_results(self, result: scipy.optimize.OptimizeResult):
        """
        Extract ScipyOptimize results into OptimizerResult.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            result from ScipyOptimizer
        """
        self.result.x = result.x
        self.result.success = result.success
        self.result.message = result.message
        self.result.fun = result.fun
        self.result.jac = result.jac
        try:
            self.result.hess = result.hess
            self.result.hess_inv = result.hess_inv
            self.result.n_iters = result.nit
            self.result.max_cv = result.maxcv
        except AttributeError:
            pass
