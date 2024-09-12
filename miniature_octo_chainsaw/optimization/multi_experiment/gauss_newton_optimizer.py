import autograd.numpy as np
from autograd import jacobian
import scipy
from typing import Tuple
from miniature_octo_chainsaw.optimization.multi_experiment.base_optimizer import BaseMultiExperimentOptimizer
from miniature_octo_chainsaw.logging_ import logger


class MultiExperimentGaussNewton(BaseMultiExperimentOptimizer):
    def __init__(
            self,
            x0: np.ndarray,
            f1_fun: callable,
            f2_fun: callable,
            n_local: int,
            n_global: int,
            n_observables: int,
            n_experiments,
            xtol: float = 1e-4,
            max_iters: int = 100,
            plot_iters: bool = False,
            compute_ci=False
    ):
        """
        Solve multi-experiment non-linear optimization problem using Gauss-Newton method.

        Schlöder, Johannes P. "Numerische Methoden zur Behandlung hochdimensionaler
        Aufgaben der Parameteridentifizierung." (Disseration) (1987).

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

        assert self.n_global > 0, (
            "No global parameters found. "
            "Multi-experiment PE does not make sense in this case!"
        )

        self.j1_fun = jacobian(self.f1_fun)
        self.j2_fun = jacobian(self.f2_fun)

        self.n_total_parameters = self.n_experiments * self.n_local + self.n_global

        y0 = self.split_into_experiments(x0)
        self.n_local_constr = len(self.f2_fun(y0[0]))

        self.n_local_rows = self.n_local_constr + self.n_observables

        self.R = None
        self.P = None
        self.G = None
        self.non_empty_rows = None

        self.j1 = np.array([]).reshape(0, self.n_total_parameters)
        self.j2 = np.array([]).reshape(0, self.n_total_parameters)

        self.T_alpha_inv = None
        self.T_beta_inv = None
        self.P_inv = None

    def solve_linearized_system(self):
        """
        Solve the linearized system using Gauss-Newton method.

        Returns
        -------
        np.ndarray : solution vector
        """
        self.T_alpha_inv, P_right = self._local_to_upper_triangular_operator(J=self.J)
        J = self.T_alpha_inv @ self.J @ P_right

        self.P_inv = self._permute_rows(J=J)
        J = self.P_inv @ J

        self.T_beta_inv = self._transform_global_blocks(J=J)
        J = self.T_beta_inv @ J

        f = self.T_beta_inv @ self.P_inv @ self.T_alpha_inv @ self.f

        Pg = self.P[-1]
        Rg = self.R[-1]
        f_global = f[-len(self.empty_rows):][:Rg.shape[1]]
        dx_global = scipy.linalg.solve(Rg @ Pg, -f_global)

        dx_local = np.zeros((self.n_experiments, self.n_local))
        k = 0
        self.G = [np.array([])] * self.n_experiments
        for i in range(self.n_experiments):
            local_rows = np.where(self.R[i].any(axis=1))[0]
            idx = slice(k, k + len(local_rows))
            self.G[i] = J[idx, -self.n_global:]

            rhs = -(self.G[i] @ dx_global + f[idx])

            dx_local[i, :] = scipy.linalg.solve((self.R[i] @ self.P[i])[local_rows], rhs)
            k += len(local_rows)

        dxbar = np.concatenate((dx_local.flatten(), dx_global))

        assert np.allclose(self.j2 @ dxbar + self.f2, 0)

        return dxbar

    def _local_to_upper_triangular_operator(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the local blocks of the Jacobian matrix into upper triangular form.

        Parameters
        ----------
        J : np.ndarray
            Jacobian matrix

        Returns
        -------
        np.ndarray : orthogonal matrix
        np.ndarray : permutation matrix
        """

        transformer = (self._transform_local_block(i=i, J=J) for i in range(self.n_experiments))
        T_, self.R, self.P = map(list, zip(*transformer))

        T_ = scipy.linalg.block_diag(*T_)
        T_inv = np.linalg.inv(T_)

        Pg = np.eye(self.n_global)
        Pl = scipy.linalg.block_diag(*self.P).T
        P = scipy.linalg.block_diag(Pl, Pg)
        return T_inv, P

    def _transform_local_block(self, i: int, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform a local block of the Jacobian matrix into upper triangular form.

        Parameters
        ----------
        i : int
            experiment index
        J : np.ndarray
            Jacobian matrix

        Returns
        -------
        np.ndarray : orthogonal matrix T
        np.ndarray : upper triangular matrix R
        np.ndarray : permutation matrix P
        """
        row_idx = slice(i * self.n_local_rows, (i + 1) * self.n_local_rows)
        J_local = J[row_idx, i * self.n_local:(i + 1) * self.n_local]
        f2, f1 = J_local[:self.n_local_constr, :], J_local[self.n_local_constr:, :]

        _, _, P = self._qr_decomposition(f2, pivoting=True)
        f2_tilde, f1_tilde = f2 @ P.T, f1 @ P.T

        T_, R11, _ = self._qr_decomposition(f2_tilde[:, :self.n_local_constr], pivoting=False)
        R12 = T_.T @ f2_tilde[:, -1:]
        L = f1_tilde[:, :self.n_local_constr] @ self._upper_triangular_inverse(R11)
        Q, R31, _ = self._qr_decomposition(f1_tilde[:, self.n_local_constr:] - L @ R12, pivoting=False)
        O = np.zeros((self.n_observables, self.n_local_constr))
        R = np.block([[R11, R12], [O, R31]])
        T_ = np.block([[T_, O.T], [L, Q]])

        assert np.allclose(T_ @ R, J_local @ P.T), f"Decomposition in {i}th block is inconsistent!"
        return T_, R, P

    def _permute_rows(self, J: np.ndarray) -> np.ndarray:
        """
        Permute the rows of the Jacobian matrix.

        Parameters
        ----------
        J : np.ndarray
            Jacobian matrix

        Returns
        -------
        np.ndarray : permutation matrix
        """
        n_rows = J.shape[0]

        where_empty = []
        for i in range(self.n_experiments):
            local_rows = np.where(~self.R[i].any(axis=1))[0]
            where_empty.append(local_rows + i * self.n_local_rows)

        self.empty_rows = np.concatenate(where_empty)
        self.non_empty_rows = np.setdiff1d(np.arange(n_rows), self.empty_rows)

        P = np.eye(n_rows)[np.concatenate((self.non_empty_rows, self.empty_rows))]
        return P

    def _transform_global_blocks(self, J: np.ndarray) -> np.ndarray:
        """
        Transform the global blocks of the Jacobian matrix into upper triangular form.

        Parameters
        ----------
        J : np.ndarray
            Jacobian matrix

        Returns
        -------
        np.ndarray : orthogonal matrix
        """
        n_empty = len(self.empty_rows)
        n_non_empty = len(self.non_empty_rows)
        assert n_empty > 0

        J_tilde = J[-n_empty:, -self.n_global:]
        _, _, Pg = self._qr_decomposition(J_tilde, pivoting=True)
        self.P.append(Pg)

        Tg, Rg, _ = self._qr_decomposition(J_tilde @ Pg.T, pivoting=False)
        self.R.append(Rg[:Rg.shape[1], :])

        T_ = scipy.linalg.block_diag(np.eye(n_non_empty), Tg)
        T_inv = np.linalg.inv(T_)

        return T_inv

    def _compute_omega(self, t: float, x: np.ndarray, dx: np.ndarray):
        f = self._function_evaluation(x + t * dx)
        f_ = self.T_beta_inv @ self.P_inv @ self.T_alpha_inv @ f

        f_global = f_[-len(self.empty_rows):][:self.Rg.shape[1]]
        dx_global = scipy.linalg.solve(self.Rg @ self.Pg, -f_global)

        dx_local = np.zeros((self.n_experiments, self.n_local))
        k = 0
        for i in range(self.n_experiments):
            local_rows = np.where(self.R[i].any(axis=1))[0]
            idx = slice(k, k + len(local_rows))

            rhs = -(self.G[i] @ dx_global + f_[idx])
            dx_local[i, :] = scipy.linalg.solve((self.R[i] @ self.P[i])[local_rows], rhs)
            k += len(local_rows)

        dxbar = np.concatenate((dx_local.flatten(), dx_global))

        numerator = 2 * np.linalg.norm(dxbar - (1 - t) * dx)
        denominator = (t ** 2) * (np.linalg.norm(dx) ** 2)
        return numerator / denominator

    def _rmt_condition(self, t: float, eta: float, x: np.ndarray, dx: np.ndarray):
        omega = self._compute_omega(t=t, x=x, dx=dx)
        return t * omega * np.linalg.norm(dx) - eta, omega

    def compute_covariance_matrix(self) -> np.ndarray:
        """
        Compute the covariance matrix at the solution.

        Returns
        -------
        np.ndarray : covariance matrix
        """
        assert len(self.P) == self.n_experiments + 1
        assert len(self.R) == self.n_experiments + 1

        C = np.zeros((self.n_total_parameters, self.n_total_parameters))

        Pg = self.P[-1]
        Rg = self.R[-1]
        Rg_inv = self._upper_triangular_inverse(Rg)
        Ug = Rg_inv @ Rg_inv.T
        C[-self.n_global:, -self.n_global:] = Pg.T @ Ug @ Pg

        for i in range(self.n_experiments):
            i_idx = slice(i * self.n_local, (i + 1) * self.n_local)
            Ri = self.R[i][:self.n_local, :self.n_local]
            Ri_inv = self._upper_triangular_inverse(Ri)
            for j in range(i, self.n_experiments):
                if i == j:
                    I = np.eye(self.n_local)
                    X = self.G[i] @ Ug @ self.G[i].T + I
                    C[i_idx, i_idx] = self.P[i].T @ Ri_inv @ X @ Ri_inv.T @ self.P[i]
                else:
                    j_idx = slice(j * self.n_local, (j + 1) * self.n_local)
                    Rj = self.R[j][:self.n_local, :self.n_local]
                    Rj_inv = self._upper_triangular_inverse(Rj)
                    X = self.G[i] @ Ug @ self.G[j].T
                    C[i_idx, j_idx] = self.P[i].T @ Ri_inv @ X @ Rj_inv.T @ self.P[j]

            C[i_idx, -self.n_global:] = self.P[i].T @ Ri_inv @ self.G[i] @ Ug @ Pg

        return C

    @staticmethod
    def _upper_triangular_inverse(x: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of an upper triangular matrix.

        Parameters
        ----------
        x : np.ndarray
            upper triangular matrix

        Returns
        -------
        np.ndarray : inverse of the upper triangular matrix
        """
        return scipy.linalg.lapack.dtrtri(x, lower=0)[0]

    @staticmethod
    def _qr_decomposition(X: np.ndarray, pivoting: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the QR decomposition of a matrix.

        Parameters
        ----------
        X : np.ndarray
            matrix
        pivoting : bool
            whether to use pivoting

        Returns
        -------
        np.ndarray : orthogonal matrix Q
        np.ndarray : upper triangular matrix R
        np.ndarray : permutation matrix P
        """
        n = X.shape[1]

        if pivoting:
            Q, R, P = scipy.linalg.qr(X, mode="full", pivoting=True)
            P = np.eye(n)[P]
        else:
            Q, R = scipy.linalg.qr(X, mode="full", pivoting=False)
            P = np.eye(n)

        return Q, R, P
