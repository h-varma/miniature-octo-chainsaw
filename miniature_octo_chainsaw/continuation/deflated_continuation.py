import copy
from typing import Union, List
import autograd.numpy as np
from autograd import jacobian
import scipy
from scipy.spatial.distance import cdist
from ..continuation.base_continuer import Continuer
from ..logging_ import logger

rng = np.random.default_rng(0)


class DeflatedContinuation(Continuer):
    def __init__(
        self,
        func: callable,
        x0: np.ndarray,
        lb: Union[np.ndarray, List[float]] = None,
        ub: Union[np.ndarray, List[float]] = None,
        p0: float = np.nan,
        p_min: float = 0,
        p_max: float = np.inf,
        p_step: float = 1,
        p_idx: int = None,
        max_failed_attempts: int = 3,
        local_optimizer: str = "scipy",
    ):
        """
        Initialize the deflated continuation method.

        Parameters
        ----------
        func : callable
            function of x and p
        x0 : np.ndarray
            initial guess
        lb : Union[np.ndarray, list[float]]
            lower bounds
        ub : Union[np.ndarray, list[float]]
            upper bounds
        p0 : float
            initial value of the parameter
        p_min : float
            minimum value of the parameter
        p_max : float
            maximum value of the parameter
        p_step : float
            step size of the parameter
        p_idx : int
            index of the parameter in the input to `func`
        max_failed_attempts : int
            maximum number of failed attempts with deflation
        local_optimizer : str
            local optimizer method for corrector step
        """
        super().__init__(
            func=func,
            x0=x0,
            lb=lb,
            ub=ub,
            p0=p0,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            p_idx=p_idx,
            local_optimizer=local_optimizer,
        )

        self.max_failed_attempts = max_failed_attempts
        self.bifurcations_found = False

        self.jacobian_ = jacobian(self.func)

        if self.p_idx is None:
            self.p_idx = len(x0)

        self._parameters = None
        self._solutions = None

        self._compute_solutions(direction=1)
        forward_solutions = copy.deepcopy(self._solutions)
        forward_parameters = copy.deepcopy(self._parameters)

        self._compute_solutions(direction=-1)
        backward_solutions = copy.deepcopy(self._solutions)
        backward_parameters = copy.deepcopy(self._parameters)

        self.parameters = backward_parameters[::-1] + forward_parameters
        self.solutions = backward_solutions[::-1] + forward_solutions

        self._remove_empty_lists()
        self._sort_the_solutions()

    def _compute_solutions(self, direction: int):
        """
        Find solutions in the given direction.

        Parameters
        ----------
        direction : int
            direction of continuation (1 for forward, -1 for backward)
        """
        x = [self.x0]
        p = np.copy(self.p0)
        p_min = copy.deepcopy(self.p0) if direction == 1 else self.p_min
        p_max = copy.deepcopy(self.p0) if direction == -1 else self.p_max

        self._parameters = [p]
        self._solutions = [[]]

        solutions = copy.deepcopy(x)

        while p_min <= p <= p_max:

            # look for new (disconnected) branches
            logger.debug(f"Deflating at parameter value: {p}.")
            for solution in solutions:
                failed_attempts = 0
                success = True
                while success or failed_attempts < self.max_failed_attempts:
                    x0 = self._add_noise(solution)
                    sol = self._deflation_step(x0=x0, p=p)
                    if sol is None:
                        success = False
                        failed_attempts += 1
                    else:
                        success = True
                        self._solutions[-1].append(sol)

            solutions = copy.deepcopy(self._solutions[-1])
            count_ = len(solutions)
            logger.debug(f"Found {count_} solutions at parameter value: {p}.")

            self._solutions.append([])

            # continue existing branches
            for solution in solutions:
                sol = self._continuation_step(x0=solution, p0=p, direction=direction)
                if sol is not None:
                    self._solutions[-1].append(sol)

            p = p + direction * self.p_step
            self._parameters.append(p)

            if len(self._solutions[-1]):
                solutions = copy.deepcopy(self._solutions[-1])
                count_ = len(solutions)
                logger.debug(f"Continued {count_} solutions to parameter value: {p}.")

    def _continuation_step(self, x0: np.ndarray, p0: float, direction: int):
        """
        Perform a continuation step using a simple predictor-corrector method.

        Parameters
        ----------
        x0 : np.ndarray
            current solution
        p0 : float
            current parameter value
        direction : int
            continuation direction (1 for forward, -1 for backward)

        Returns
        -------
        np.ndarray : new solution
        """

        # predictor step
        sol = self._join_x_vector_and_p(x0, p0)
        Jx, Jp = self._compute_jacobians(sol)

        step_vector = self._solve_linear_system(Jx, -Jp)
        x1 = x0 + direction * self.p_step * step_vector
        p1 = p0 + direction * self.p_step

        # corrector step
        def corrector(_x):
            _sol = self._join_x_vector_and_p(_x, p1)
            return self.func(_sol)

        x, _ = self._solve_optimization_problem(corrector, x0=x1, lb=self.lb, ub=self.ub)
        if x is None:
            return None

        is_in_history = self._check_if_solution_already_exists(x)
        return None if is_in_history else x

    def _deflation_step(self, x0: np.ndarray, p: float) -> np.ndarray:
        """
        Perform a deflation step to find a new solution.

        Parameters
        ----------
        x0 : np.ndarray
            initial guess for the solution
        p : float
            parameter value

        Returns
        -------
        np.ndarray : new solution at parameter value p
        """
        known_solutions = self._solutions[-1]

        def _deflated_corrector(_x):
            sol = self._join_x_vector_and_p(_x, p)
            df = self.func(sol)
            for solution in known_solutions:
                df = np.dot(self._deflation_operator(_x, solution), df)
            return df

        x, _ = self._solve_optimization_problem(_deflated_corrector, x0=x0, lb=self.lb, ub=self.ub)
        if x is None:
            return None

        is_in_history = self._check_if_solution_already_exists(x)
        is_valid = self._check_if_solution_satisfies_ftol(x=x, p=p, ftol=1e-6)
        return x if is_valid and not is_in_history else None

    @staticmethod
    def _deflation_operator(u: np.ndarray, ustar: np.ndarray) -> np.ndarray:
        """
        Operator to deflate out known solutions.

        Parameters
        ----------
        u : np.ndarray
            current solution
        ustar : np.ndarray
            known solution

        Returns
        -------
        np.ndarray : deflation operator value
        """
        return (1 + (1 / np.sum((u - ustar) ** 2))) * np.eye(len(u))

    @staticmethod
    def _add_noise(x: np.ndarray) -> np.ndarray:
        """
        Add normally distributed noise to the input data.

        Parameters
        ----------
        x : np.ndarray
            input data

        Returns
        -------
        np.ndarray : input data with added noise
        """
        mean = x
        stddev = np.maximum(1, np.abs(x))

        a, b = -mean / stddev, np.inf
        return scipy.stats.truncnorm.rvs(a=a, b=b, loc=mean, scale=stddev)

    def _check_if_solution_already_exists(self, x: np.ndarray) -> bool:
        """
        Check if `x` is the same as a previously found result, differing only in their signs.

        Parameters
        ----------
        x : np.ndarray
            solution to check

        Returns
        -------
        bool : True if the solution already exists, False otherwise
        """
        flag = [np.allclose(np.abs(x), np.abs(s), atol=0) for s in self._solutions[-1]]
        return True if True in flag else False

    def _sort_the_solutions(self):
        """
        Sort steady states at a parameter value to continuously follow
        steady states from the previous parameter value.
        """
        self.solutions[0] = np.row_stack(self.solutions[0])
        max_size = self.solutions[0].shape[0]

        for i in range(1, len(self.solutions)):
            self.solutions[i] = np.row_stack(self.solutions[i])
            dist_ = cdist(self.solutions[i], self.solutions[i - 1])

            previous_size = self.solutions[i - 1].shape[0]
            current_size = self.solutions[i].shape[0]
            if current_size > max_size:
                max_size = current_size

            for _ in range(previous_size):
                if np.all(np.isnan(dist_)):
                    break
                row, col = np.argwhere(dist_ == np.nanmin(dist_))[0]
                dist_[:, col] = np.nan
                dist_[row, :] = np.nan

                if current_size >= previous_size and row != col:
                    self.solutions[i][[row, col]] = self.solutions[i][[col, row]]
                    dist_[[row, col]] = dist_[[col, row]]
                    if i == 1:
                        self.solutions[0] = self._insert_nan_rows(self.solutions[0], row)

                elif current_size < previous_size and row != col:
                    self.solutions[i] = self._insert_nan_rows(self.solutions[i], row)
                    dist_ = self._insert_nan_rows(dist_, row)

        for i in range(len(self.solutions)):
            if self.solutions[i].shape[0] < max_size:
                idx = np.arange(self.solutions[i].shape[0], max_size)
                for j in idx:
                    self.solutions[i] = self._insert_nan_rows(self.solutions[i], j)

    @staticmethod
    def _insert_nan_rows(x: np.ndarray, idx: Union[int, list, slice]) -> np.ndarray:
        """
        Insert rows filled with nan into the input matrix.

        Parameters
        ----------
        x : np.ndarray
            input matrix
        idx : int
            index at which to insert nan rows

        Returns
        -------
        np.ndarray : matrix with nan rows inserted
        """
        return np.insert(x, idx, np.nan, axis=0)

    @staticmethod
    def _size(X: np.ndarray) -> int:
        """
        Compute the number of non-nan rows in a matrix.

        Parameters
        ----------
        X : np.ndarray
            input matrix

        Returns
        -------
        int : number of non-nan rows
        """
        return len([X[i, :] for i in range(X.shape[0]) if not any(np.isnan(X[i, :]))])

    def _remove_empty_lists(self):
        """
        Remove empty solutions lists from the solutions and parameters.
        """
        idx = []
        for i, sol in enumerate(self.solutions):
            if len(sol) == 0:
                idx.append(i)
        self.parameters = [p for i, p in enumerate(self.parameters) if i not in idx]
        self.solutions = [s for i, s in enumerate(self.solutions) if i not in idx]

    def detect_saddle_node_bifurcation(self, parameter: str) -> np.ndarray:
        """
        Detect saddle-node bifurcation branches in the solutions.

        Parameters
        ----------
        parameter : str
            name of bifurcation parameter

        Returns
        -------
        np.ndarray : saddle-node bifurcation point
        """

        branches = [[]]
        old = self._join_x_matrix_and_p(x=self.solutions[0], p=self.parameters[0])
        for i in range(1, len(self.solutions)):
            new = self._join_x_matrix_and_p(x=self.solutions[i], p=self.parameters[i])
            change_in_solutions = self._size(new) - self._size(old)

            if 1 <= change_in_solutions <= 2:
                for j in range(new.shape[0]):
                    if np.isnan(old[j]).any() and not np.isnan(new[j]).any():
                        branches[-1].append(new[j])

            elif -2 <= change_in_solutions <= -1:
                for j in range(old.shape[0]):
                    if np.isnan(new[j]).any() and not np.isnan(old[j]).any():
                        branches[-1].append(old[j])

            if len(branches[-1]) == 2:
                branches[-1] = sum(branches[-1]) / 2
                self.bifurcations_found = True
                branches.append([])

            old = new

        return self._select_bifurcation_point(branches=branches, parameter=parameter)

    def detect_hopf_bifurcation(self, parameter: str) -> np.ndarray:
        """
        Detect Hopf bifurcation points in the solutions.

        Parameters
        ----------
        parameter : str
            name of bifurcation parameter

        Returns
        -------
        np.ndarray : Hopf bifurcation point
        """
        old_eigvals = self._get_eigenvalues(self.solutions[0], self.parameters[0])
        old_signs = np.sign(old_eigvals.real)

        branches = [[]]
        for i in range(1, len(self.solutions)):
            new_eigvals = self._get_eigenvalues(self.solutions[i], self.parameters[i])
            new_signs = np.sign(new_eigvals.real)

            if (old_signs != new_signs).any():
                sign_change = np.abs(old_signs - new_signs) == 2
                is_complex = np.iscomplex(old_eigvals) | np.iscomplex(new_eigvals)
                mask = np.any(sign_change & is_complex, axis=1)
                solution = self.solutions[i][mask, :]
                if len(solution):
                    solution = self._join_x_vector_and_p(solution, self.parameters[i])
                    branches[-1].extend(solution)
                    self.bifurcations_found = True
                    branches.append([])

            old_eigvals = new_eigvals
            old_signs = new_signs

        return self._select_bifurcation_point(branches=branches, parameter=parameter)

    def _get_eigenvalues(self, solutions: np.ndarray, parameter: float) -> np.ndarray:
        """
        Compute the eigenvalues of the Jacobian matrix at each solution in solutions.

        Parameters
        ----------
        solutions : np.ndarray
            solutions at which to compute the eigenvalues

        Returns
        -------
        np.ndarray : eigenvalues of the Jacobian matrix
        """
        eigenvalues = []
        solutions = self._join_x_matrix_and_p(x=solutions, p=parameter)
        n_solutions, n_variables = solutions.shape
        for i in range(n_solutions):
            if np.any(np.isnan(solutions[i, :])):
                eigenvalues.append(np.nan * np.ones(n_variables - 1))
            else:
                jacobian_ = self.jacobian_(solutions[i, :])
                jacobian_ = np.delete(jacobian_, self.p_idx, axis=1)
                eigenvalues.append(np.linalg.eigvals(jacobian_))
        return np.row_stack([eigenvalues])

    def _select_bifurcation_point(self, branches: list, parameter: str) -> np.ndarray:
        """
        Select one bifurcation point from the detected bifurcation points.

        Parameters
        ----------
        branches : list
            bifurcation branches
        parameter : str
            parameter name

        Returns
        -------
        np.ndarray : bifurcation point
        """
        branches = [branch for branch in branches if len(branch) > 0]

        if not self.bifurcations_found:
            logger.error("No bifurcations could be found in the given parameter range!")
            raise ValueError(
                "No bifurcations could be found in the given parameter range. "
                "Try using different initial guesses or expand the parameter range."
            )
        elif len(branches) == 1:
            logger.info(f"A bifurcation was detected near {parameter} = {branches[0][self.p_idx]}.")
            return branches[0]
        else:
            logger.info(f"Bifurcations were detected near the following values of {parameter}:")
            for i, branch in enumerate(branches):
                logger.info(f"{i + 1}: {branch[self.p_idx]}")
            idx = int(input("Select a bifurcation point for continuation: "))
            return branches[idx - 1]
