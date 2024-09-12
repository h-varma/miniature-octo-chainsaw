import autograd.numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.stats import chi2
from miniature_octo_chainsaw.optimization.single_experiment.base_optimizer import BaseOptimizer
from miniature_octo_chainsaw.optimization.check_regularity import check_CQ, check_PD
from miniature_octo_chainsaw.optimization.multi_experiment.line_search import line_search
from miniature_octo_chainsaw.logging_ import logger


class BaseMultiExperimentOptimizer(BaseOptimizer, ABC):
    """
    Multi-experiment optimizer interface - not functional on its own.
    """

    def __init__(self):
        """
        Initialize the optimizer.
        """
        super().__init__()

        self.f = None
        self.J = None

        self.f1 = np.array([])
        self.f2 = np.array([])
        self.lagrange_multipliers = None

    def minimize(self, line_search_strategy: str = "exact"):
        """
        Minimize the objective function subject to equality constraints.

        Parameters
        ----------
        line_search_strategy : str
            name of the line search strategy
        """
        x = np.copy(self.x0)

        for i in range(self.max_iters):
            self.f = self._function_evaluation(x=x)
            self.J = self._jacobian_evaluation(x=x)

            if not check_PD(j=self.J):
                logger.warn(f"Positive definiteness does not hold in iterate {i}!")

            dxbar = self.solve_linearized_system()

            t = line_search(
                x=x,
                dx=dxbar,
                func=self._level_function,
                strategy=line_search_strategy,
            )
            dx = dxbar * t

            if np.linalg.norm(dx) < self.xtol:
                self.result.success = True
                self.result.message = "Parameter estimation solver converged!"
                break

            x = x + dx

            self.result.x = x
            level_function = self._level_function(x=x)
            self.result.func = level_function
            self.result.level_functions.append(level_function)
            self.result.n_iters = i + 1

            if i == self.max_iters - 1:
                self.result.message = "Maximum number of iterations reached!"
                logger.warn(self.result.message)

        if self.plot_iters:
            self._plot_iterations()

        if self.compute_ci:
            self.result.covariance_matrix = self.compute_covariance_matrix(x)
            self.result.confidence_intervals = self._compute_confidence_intervals(x)

    @abstractmethod
    def solve_linearized_system(self) -> np.ndarray:
        """
        Solve the linearized system.

        Returns
        -------
        np.ndarray : solution vector
        """
        raise NotImplementedError

    @abstractmethod
    def compute_covariance_matrix(self, x: np.ndarray = None) -> np.ndarray:
        """
        Compute the covariance matrix.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : covariance matrix
        """
        raise NotImplementedError

    def _compute_confidence_intervals(
            self,
            x: np.ndarray,
            significance: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals for solution vector.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        significance : float
            significance level of the confidence intervals

        Returns
        -------
        np.ndarray : confidence intervals
        """
        C = self.compute_covariance_matrix(x)
        Cii = np.diag(C)

        self._function_evaluation(x)
        n_objective = len(self.f1)
        n_constraints = len(self.f2)

        # compute common factor 'beta' (see Schlöder1988 or Natermann's Diss)
        factor = n_objective + n_constraints - self.n_total_parameters
        beta = np.linalg.norm(self.f1) / np.sqrt(factor)

        # compute quantile of chi2 distribution (see Körkel's Diss or Bard's Book)
        dof = self.n_total_parameters - n_constraints
        gamma = np.sqrt(chi2.ppf(significance, dof))

        return beta * gamma * np.sqrt(Cii)

    def _plot_iterations(self):
        """Plot the level function at each iteration."""
        iterations = np.arange(0, len(self.result.level_functions))
        function_values = self.result.level_functions
        plt.plot(iterations, function_values, marker="o", color="black")
        plt.xlabel("number of iterations")
        plt.ylabel("level function value")
        plt.show()

    def split_into_experiments(self, x: np.ndarray) -> np.ndarray:
        """
        Split the solution vector into local experiments with common global parameters.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : solution matrix with experiments as rows
        """
        local_x = x[: -self.n_global]
        local_x = local_x.reshape(self.n_experiments, self.n_local)

        global_x = x[-self.n_global:]
        global_x = global_x.reshape(-1, 1)
        global_x = np.tile(global_x, self.n_experiments)

        return np.column_stack((local_x, global_x.T))

    def _function_evaluation(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective and constraints.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : function values
        """
        f = np.array([])

        self.f1 = self.f1_fun(x)
        self.f2 = np.array([])

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            f1 = self.f1[i * self.n_observables: (i + 1) * self.n_observables]
            f2 = self.f2_fun(x[i])
            f = np.concatenate((f, f2, f1))
            self.f2 = np.concatenate((self.f2, f2))

        return f

    def _jacobian_evaluation(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Jacobian of the objective and constraints.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : Jacobian matrix
        """
        n_cols = self.n_total_parameters
        J = np.array([]).reshape(0, n_cols)

        self.j1 = self.j1_fun(x)
        self.j2 = np.array([]).reshape(0, n_cols)

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            j1 = self.j1[i * self.n_observables: (i + 1) * self.n_observables]

            n_rows = self.j2_fun(x[i]).shape[0]
            j2 = np.zeros((n_rows, n_cols))
            j2_ = self.j2_fun(x[i])
            local_idx = slice(i * self.n_local, (i + 1) * self.n_local)
            global_idx = slice(self.n_experiments * self.n_local, None)
            j2[:, local_idx] = j2_[:, : self.n_local]
            j2[:, global_idx] = j2_[:, self.n_local:]
            assert check_CQ(j2), f"Experiment {i}: No constraint qualification!"

            J = np.row_stack((J, j2, j1))
            self.j2 = np.row_stack((self.j2, j2))
        assert check_CQ(self.j2), "Constraint qualification does not hold!"

        return J

    def _level_function(self, x: np.ndarray) -> float:
        """
        Compute the value of the level function.

        Parameters
        ----------
        x : np.ndarray
            value of the current iterate

        Returns
        -------
        float : value of the level function
        """
        objective = 0.5 * np.linalg.norm(self.f1_fun(x), ord=2) ** 2

        x = self.split_into_experiments(x)
        lagrange_multipliers = self.lagrange_multipliers.reshape(self.n_experiments, -1)

        constraints = np.zeros(self.n_experiments)
        for i in range(self.n_experiments):
            f2 = np.linalg.norm(self.f2_fun(x[i]), ord=1)
            beta = np.linalg.norm(lagrange_multipliers[i, :], ord=np.inf)
            constraints[i] = beta * f2

        return objective + np.sum(constraints)
