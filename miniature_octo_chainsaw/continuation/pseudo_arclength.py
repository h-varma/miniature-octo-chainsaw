import copy
from bisect import bisect, bisect_left
import autograd.numpy as np
from ..continuation.base_continuer import Continuer
from ..logging_ import logger


class PseudoArclengthContinuation(Continuer):
    def __init__(
        self,
        func: callable,
        x0: np.ndarray,
        p0: float = np.nan,
        p_min: float = -np.inf,
        p_max: float = np.inf,
        p_step: float = 1,
        p_step_min: float = 1e-6,
        p_step_max: float = np.inf,
        p_idx: int = -1,
        max_iters: int = 1000,
        max_newton_iters: int = 10,
        newton_tol: float = 1e-4,
        fast_iters: int = 3,
        data: np.ndarray = None,
    ):
        """
        Initialize the deflated continuation method.

        Parameters
        ----------
        func : callable
            function of x and p
        x0 : np.ndarray
            initial guess
        p0 : float
            initial value of the parameter
        p_min : float
            minimum value of the parameter
        p_max : float
            maximum value of the parameter
        p_step : float
            step size of the parameter
        p_step_min : float
            minimum step size of the parameter
        p_idx : int
            index of the parameter in the input to `func`
        max_iters : int
            maximum number of predictor-corrector iterations
        max_newton_iters : int
            maximum number of iterations for newton-corrector
        newton_tol : float
            tolerance for newton-corrector
        fast_iters : int
            number of optimizer iterations for fast convergence
        data : np.ndarray
            data points to trace
        """
        super().__init__(
            func=func,
            x0=x0,
            p0=p0,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            p_idx=p_idx,
        )

        self.p_step_min = p_step_min
        self.p_step_max = p_step_max
        self.max_iters = max_iters
        self.max_newton_iters = max_newton_iters
        self.newton_tol = newton_tol
        self.fast_iters = fast_iters
        self.data = data

        if self.p_idx is None:
            self.p_idx = len(x0)

        self._parameters = None
        self._solutions = None

        self._direction_str = {1: "forward", -1: "backward"}
        self._bisect_funcs = {1: bisect, -1: bisect_left}
        self.flag = False

        self._compute_solutions(direction=1)
        forward_solutions = copy.deepcopy(self._solutions)
        forward_parameters = copy.deepcopy(self._parameters)

        self._compute_solutions(direction=-1)
        backward_solutions = copy.deepcopy(self._solutions)
        backward_parameters = copy.deepcopy(self._parameters)

        self.parameters = backward_parameters[::-1] + forward_parameters
        self.solutions = backward_solutions[::-1] + forward_solutions

    def _compute_solutions(self, direction: int):
        """
        Find solutions in the given direction.

        Parameters
        ----------
        direction : int
            direction of continuation (1 for forward, -1 for backward)
        """
        y = self._join_x_vector_and_p(x=self.x0, p=self.p0)

        p_min = self.p_min
        p_max = float(np.maximum(self.p0, self.p_max))
        step = self.p_step

        self._parameters = []
        self._solutions = []

        Jx, Jp = self._compute_jacobians(y)
        step_vector = self._solve_linear_system(A=Jx, b=-Jp)

        for i in range(self.max_iters):
            p = y[self.p_idx]
            x = np.delete(y, self.p_idx)

            dp = direction / np.sqrt(1 + (np.linalg.norm(step_vector) ** 2))
            dx = step_vector * dp
            success = False
            while not success and step >= self.p_step_min:
                if self.data is not None:
                    step = self._trace_data(x=p, dx=dp, step=step, direction=direction)

                x_, p_, step, success = self._corrector_step(x0=x, dx0=dx, p0=p, dp0=dp, step=step)

                if success is True:
                    x, p = x_.copy(), p_

            if p < p_min or p > p_max or step < self.p_step_min:
                break

            logger.debug(
                f"Continued solution to parameter value: {p} in {self._direction_str[direction]} direction."
            )
            self._parameters.append(p)
            self._solutions.append(x)

            y = self._join_x_vector_and_p(x=x, p=p)
            Jx, Jp = self._compute_jacobians(y)
            step_vector = self._solve_linear_system(A=Jx, b=-Jp)
            if np.sign(dx.T @ step_vector + dp) != direction:
                old_direction = self._direction_str[direction]
                new_direction = self._direction_str[np.sign(dx.T @ step_vector + dp)]
                logger.debug(f"Changing directions: {old_direction} -> {new_direction}")
            direction = np.sign(dx.T @ step_vector + dp)

    def _trace_data(self, x: np.ndarray, dx: np.ndarray, step: float, direction: int) -> float:
        """
        Adjust step size to trace measurements.

        Parameters
        ----------
        x : np.ndarray
            initial value
        dx : np.ndarray
            step vector
        step : float
            step size
        """

        _bisect = self._bisect_funcs[direction]
        idx1 = _bisect(self.data, x)
        idx2 = _bisect(self.data, x + step * dx)
        if idx1 < idx2:
            x_tilde = self.data[idx1]
        elif idx1 > idx2:
            x_tilde = self.data[idx1 - 1]
        else:
            self.flag = False
            return step
        self.flag = True
        return (x_tilde - x) / dx

    def _corrector_step(self, x0: np.ndarray, dx0: np.ndarray, p0: float, dp0: float, step: float):
        """
        Perform a corrector step to find a solution.

        Parameters
        ----------
        x0 : np.ndarray
            initial guess for x
        dx0 : np.ndarray
            step vector for x
        p0 : float
            initial guess for p
        dp0 : float
            step for p
        step : float
            step size for p

        Returns
        -------
        x : np.ndarray
            solution for x
        p : float
            solution for p
        step : float
            updated step size for p
        success : bool
            flag indicating successful correction
        """
        success = False
        x = x0 + step * dx0
        p = p0 + step * dp0

        for i in range(self.max_newton_iters):
            y = self._join_x_vector_and_p(x=x, p=p)
            Jx, Jp = self._compute_jacobians(y)

            obj_func = self.func(y)
            if self.flag:
                dx = self._solve_linear_system(A=Jx, b=-obj_func)
                dp = 0
                dy = self._join_x_vector_and_p(dx, dp)
            else:
                row1 = np.insert(Jx, self.p_idx, Jp, axis=1)
                row2 = np.insert(dx0, self.p_idx, dp0)
                coeff = np.row_stack((row1, row2))

                cont_func = (x - x0).T @ dx0 + (p - p0) * dp0 - step
                rhs = np.hstack((obj_func, cont_func))
                dy = self._solve_linear_system(A=coeff, b=-rhs)
                dx = np.delete(dy, self.p_idx)
                dp = dy[self.p_idx]

            x = x + dx
            p = p + dp

            if np.linalg.norm(dy) < self.newton_tol:
                success = True
                if step <= self.p_step_max and i < self.fast_iters:
                    step = step * 2
                break

        if not success:
            step = step / 2

        return x, p, step, success
