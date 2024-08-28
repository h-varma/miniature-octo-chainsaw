import autograd.numpy as np
from scipy.stats import chi2
from autograd import jacobian
from scipy import sparse
import matplotlib.pyplot as plt
import osqp
from miniature_octo_chainsaw.optimization.check_regularity import check_PD, check_CQ
from miniature_octo_chainsaw.optimization.multi_experiment.base_optimizer import BaseMultiExperimentOptimizer
from miniature_octo_chainsaw.optimization.multi_experiment.line_search import line_search
from miniature_octo_chainsaw.logging_ import logger


class MultiExperimentOSQP(BaseMultiExperimentOptimizer):
    def __init__(
        self,
        x0: np.ndarray,
        f1_fun: callable,
        f2_fun: callable,
        n_local: int,
        n_global: int,
        n_observables: int,
        n_experiments: int,
        xtol: float = 1e-4,
        max_iters: int = 100,
        plot_iters: bool = False,
        compute_ci: bool = False,
    ):
        """
        Solve multi-experiment non-linear optimization problem using OSQP.

        Stellato, Bartolomeo, et al. "OSQP: An operator splitting solver for quadratic programs."
        Mathematical Programming Computation 12.4 (2020): 637-672.

        Parameters
        ----------
        x0 : np.ndarray
            initial guess
        f1_fun : callable
            objective function
        f2_fun : callable
            equality constraint
        n_local : int
            number of local parameters
        n_global : int
            number of global parameters
        n_observables : int
            number of observables
        n_experiments : int
            number of experiments
        xtol : float
            convergence threshold
        max_iters : int
            maximum number of iterations
        plot_iters : bool
            whether to plot the level function at each iteration
        compute_ci : bool
            whether to compute confidence intervals
        """

        super().__init__()
        self.x0 = x0
        self.f1_fun = f1_fun
        self.f2_fun = f2_fun
        self.n_local = n_local
        self.n_global = n_global
        self.n_observables = n_observables
        self.n_experiments = n_experiments
        self.xtol = xtol
        self.max_iters = max_iters
        self.plot_iters = plot_iters
        self.compute_ci = compute_ci

        self.j1_fun = jacobian(self.f1_fun)
        self.j2_fun = jacobian(self.f2_fun)

        self.n_total_parameters = self.n_experiments * self.n_local + self.n_global

        assert self.n_global > 0, (
            "No global parameters found. "
            "Multi-experiment PE does not make sense in this case!"
        )

        self.f1 = np.array([])
        self.f2 = np.array([])
        self.j1 = np.array([])
        self.j2 = np.array([]).reshape(0, self.n_total_parameters)
        self.lagrange_multipliers = None

    def minimize(self, *kwargs):
        """Minimize the objective function subject to equality constraints."""
        x = np.copy(self.x0)

        for i in range(self.max_iters):
            self.__function_evaluation(x=x)
            self.__jacobian_evaluation(x=x)

            if not check_PD(j=self.J):
                logger.warn(f"Positive definiteness does not hold in iterate {i}!")

            dxbar = self.__solve_linearized_system_using_osqp()

            t = line_search(
                x=x,
                dx=dxbar,
                func=self.level_function,
                strategy="exact",
            )
            dx = dxbar * t

            if np.linalg.norm(dx) < self.xtol:
                self.result.success = True
                self.result.message = "Parameter estimation solver converged!"
                break

            x = x + dx

            self.result.x = x
            level_function = self.level_function(x=x)
            self.result.func = level_function
            self.result.level_functions.append(level_function)
            self.result.n_iters = i + 1

            if i == self.max_iters - 1:
                self.result.message = "Maximum number of iterations reached!"
                logger.warn(self.result.message)

        if self.plot_iters:
            self.__plot_iterations()

        if self.compute_ci:
            self.result.covariance_matrix = self.__compute_covariance_matrix(x)
            self.result.confidence_intervals = self.__compute_confidence_intervals(x)

    def __plot_iterations(self):
        """Plot the level function at each iteration."""
        iterations = np.arange(0, len(self.result.level_functions))
        function_values = self.result.level_functions
        plt.plot(iterations, function_values, marker="o", color="black")
        plt.xlabel("number of iterations")
        plt.ylabel("level function value")
        plt.show()

    def __jacobian_evaluation(self, x: np.ndarray):
        """
        Evaluate the Jacobian of the objective and constraints.

        Parameters
        ----------
        x : np.ndarray
            solution vector
        """
        n_cols = self.n_total_parameters
        self.J = np.array([]).reshape(0, n_cols)

        self.j1 = self.j1_fun(x)
        self.j2 = np.array([]).reshape(0, n_cols)

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            j1 = self.j1[i * self.n_observables : (i + 1) * self.n_observables]

            n_rows = self.j2_fun(x[i]).shape[0]
            j2 = np.zeros((n_rows, n_cols))
            j2_ = self.j2_fun(x[i])
            local_idx = slice(i * self.n_local, (i + 1) * self.n_local)
            global_idx = slice(self.n_experiments * self.n_local, None)
            j2[:, local_idx] = j2_[:, : self.n_local]
            j2[:, global_idx] = j2_[:, self.n_local :]
            assert check_CQ(j2), f"Experiment {i}: No constraint qualification!"

            self.J = np.row_stack((self.J, j2, j1))
            self.j2 = np.row_stack((self.j2, j2))
        assert check_CQ(self.j2), "Constraint qualification does not hold!"

    def __function_evaluation(self, x: np.ndarray):
        """
        Evaluate the objective and constraints.

        Parameters
        ----------
        x : np.ndarray
            solution vector
        """
        self.f = np.array([])

        self.f1 = self.f1_fun(x)
        self.f2 = np.array([])

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            f1 = self.f1[i * self.n_observables : (i + 1) * self.n_observables]
            f2 = self.f2_fun(x[i])
            self.f = np.concatenate((self.f, f2, f1))
            self.f2 = np.concatenate((self.f2, f2))

    def __solve_linearized_system_using_osqp(self) -> np.ndarray:
        """Solve the linearized system using OSQP.

        Returns
        -------
        np.ndarray : solution of the linearized system
        """
        P = sparse.csr_matrix(self.j1.T @ self.j1)
        q = self.j1.T @ self.f1
        A = sparse.csr_matrix(self.j2)
        l, u = -self.f2, -self.f2

        problem = osqp.OSQP()
        problem.setup(P, q, A, l, u)
        res = problem.solve()

        self.lagrange_multipliers = res.y.copy()
        return res.x

    def level_function(self, x: np.ndarray) -> float:
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

    def __compute_covariance_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix at the solution.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : covariance matrix
        """
        self.__jacobian_evaluation(x)

        n_con = self.j2.shape[0]
        n_var = self.j1.shape[1]

        I = np.eye(n_var)
        O1 = np.zeros((n_var, n_var))
        O2 = np.zeros((n_var, n_con))
        O3 = np.zeros((n_con, n_con))
        X = np.block([[self.j1.T @ self.j1, O2], [O2.T, O3]])
        Id = np.column_stack((I, O2))
        KKT = X + np.block([[O1, self.j2.T], [self.j2, O3]])
        KKT_inv = np.linalg.inv(KKT)

        return Id @ KKT_inv @ X @ KKT_inv.T @ Id.T

    def __compute_confidence_intervals(self, x: np.ndarray) -> np.ndarray:
        """
        Compute confidence intervals for solution vector.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : confidence intervals
        """
        C = self.__compute_covariance_matrix(x)
        Cii = np.diag(C)

        self.__function_evaluation(x)
        n_objective = len(self.f1)
        n_constraints = len(self.f2)

        # compute common factor 'beta' (see Schlöder1988 or Natermann's Diss)
        factor = n_objective + n_constraints - self.n_total_parameters
        beta = np.linalg.norm(self.f1) / np.sqrt(factor)

        # compute quantile of chi2 distribution (see Körkel's Diss or Bard's Book)
        dof = self.n_total_parameters - n_constraints
        gamma = np.sqrt(chi2.ppf(0.05, dof))

        return beta * gamma * np.sqrt(Cii)
