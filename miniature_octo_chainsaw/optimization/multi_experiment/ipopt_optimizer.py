import scipy
import autograd.numpy as np
import cyipopt
from miniature_octo_chainsaw.source.optimization.multi_experiment.base_optimizer import BaseMultiExperimentOptimizer


class MultiExperimentIpopt(BaseMultiExperimentOptimizer):
    """
    Use the IpOpt library to solve a large-scale non-linear optimization problem.

    Find details on the optimizer at https://pypi.org/project/ipopt/.
    """

    def __init__(
        self,
        x0: np.ndarray,
        f1_fun: callable,
        f2_fun: callable,
        xtol: float = 1e-4,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        x0 : np.ndarray
            initial guess
        f1_fun : callable
            objective function
        f2_fun : callable
            equality constraint
        xtol : float
            convergence threshold
        """
        super().__init__()
        self.x0 = x0
        self.f1_fun = f1_fun
        self.f2_fun = f2_fun
        self.xtol = xtol

    def minimize(self, **kwargs):
        """Minimize the objective function subject to equality constraints."""
        f1_fun = self._check_if_valid_objective(objective=self.f1_fun)

        res = cyipopt.minimize_ipopt(
            fun=f1_fun,
            x0=self.x0,
            # bounds=self.bounds,
            constraints=self.f2_fun,
            tol=self.xtol,
        )

        self._get_results(res)

    def _get_results(self, result: scipy.optimize.OptimizeResult):
        """
        Extract IpOpt results into OptimizerResult.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            result from ScipyOptimizer
        """
        self.result.x = result.x
        self.result.success = result.success
        self.result.message = result.message
        self.result.func = result.fun
        self.result.n_iters = result.nit

    def _check_if_valid_objective(self, objective: callable) -> callable:
        """
        Check if the objective function is valid for IpOpt.

        Parameters
        ----------
        objective : callable
            objective function

        Returns
        -------
        callable : (modified) objective function
        """
        is_least_squares = bool(not isinstance(objective(self.x0), float))
        if is_least_squares:
            return lambda x: np.linalg.norm(objective(x)) ** 2
        else:
            return objective
