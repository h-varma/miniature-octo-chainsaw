import autograd.numpy as np
from typing import Union, List, Tuple
from autograd import jacobian
from miniature_octo_chainsaw.optimization.single_experiment.select_optimizer import import_optimizer


class Continuer:
    """
    Continuer interface, not functional on its own.

    The continuer takes a non-linear system, a known solution and a parameter range and continues the solution.
    It returns the ContinuationSolution object.
    """

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
        p_idx: int = -1,
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
        local_optimizer : str
            local optimizer method for corrector step
        """
        self.func = func
        self.x0 = x0
        self.lb = lb
        self.ub = ub
        self.p0 = p0
        self.p_min = p_min
        self.p_max = p_max
        self.p_step = p_step
        self.p_idx = p_idx
        self.local_optimizer = local_optimizer

    def _join_x_vector_and_p(self, x: np.ndarray, p: float) -> np.ndarray:
        """
        Combine the solution vector `x` with the parameter `p`.

        Parameters
        ----------
        x : np.ndarray
            solution
        p : float
            parameter value

        Returns
        -------
        np.ndarray : combined solution
        """
        return np.insert(arr=x, obj=self.p_idx, values=p)

    def _join_x_matrix_and_p(self, x: np.ndarray, p: float) -> np.ndarray:
        """
        Combine the solution matrix `x` with the parameter `p`.

        Parameters
        ----------
        x : np.ndarray
            row-wise solution matrix
        p : float
            parameter value

        Returns
        -------
        np.ndarray : combined solution
        """
        if len(x.shape) == 1:
            x = np.row_stack([x])
        return np.insert(arr=x, obj=self.p_idx, values=p, axis=1)

    def _solve_optimization_problem(
        self,
        func: callable,
        x0: np.ndarray,
        lb: np.ndarray = None,
        ub: np.ndarray = None,
        method: str = None,
    ):
        """
        Solve the system of equations `func` to find a solution.

        Parameters
        ----------
        func : callable
            system of non-linear equations
        x0 : np.ndarray
            initial guess
        lb : np.ndarray
            lower bounds
        ub : np.ndarray
            upper bounds
        method : str
            optimization method

        Returns
        -------
        np.ndarray : solution to the system of equations
        """
        Optimizer = import_optimizer(self.local_optimizer)
        optimizer = Optimizer(func, x0=x0, lb=lb, ub=ub)
        try:
            optimizer.minimize(method=method)
            if optimizer.result.success:
                x = optimizer.result.x
                n_iters = optimizer.result.n_iters
                return x, n_iters
            else:
                return None, None
        except ZeroDivisionError:
            return None, None

    def _check_if_solution_satisfies_ftol(self, x: np.ndarray, p: float, ftol: float):
        """
        Check if the solution satisfies the stopping criterion.

        Parameters
        ----------
        x : np.ndarray
            solution
        p : float
            parameter value
        ftol : float
            tolerance value

        Returns
        -------
        bool : True if the solution satisfies the stopping criterion, False otherwise
        """
        solution = self._join_x_vector_and_p(x, p)
        return np.linalg.norm(self.func(solution)) < ftol

    def _compute_jacobians(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobians of the function with respect to x and p.

        Parameters
        ----------
        y : np.ndarray
            input to the function

        Returns
        -------
        Jx : np.ndarray
            Jacobian of the function with respect to x
        Jp : np.ndarray
            Jacobian of the function with respect to p
        """
        J = jacobian(self.func)(y)
        Jx = np.delete(J, self.p_idx, axis=1)
        Jp = J[:, self.p_idx]
        return Jx, Jp

    @staticmethod
    def _solve_linear_system(A: np.ndarray, b: np.ndarray, rcond: float = 1e-6):
        """
        Solve system of linear equations using least squares solver.

        Parameters
        ----------
        A : np.ndarray
            coefficient matrix
        b : np.ndarray
            right-hand side vector
        rcond : float
            cutoff for small singular values of A

        Returns
        -------
        np.ndarray : solution to the system of linear equations
        """
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
        return x
