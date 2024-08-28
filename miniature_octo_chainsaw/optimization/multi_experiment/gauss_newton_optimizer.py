import numpy as np
from autograd import jacobian
import scipy
import matplotlib.pyplot as plt
import logging
import copy
from miniature_octo_chainsaw.source.optimization.check_regularity import check_PD, check_CQ


CONVERGENCE_THRESHOLD = 1e-4
MAX_ITERS = 100
ETA = 1
ETA_MAX = 1.2


class MultiExperimentGaussNewton:
    def __init__(self, x0, f1_fun, f2_fun, n_experiments, settings, compute_ci=False):
        self.f1_fun = f1_fun  # objective function
        self.f2_fun = f2_fun  # equality constraint
        self.j1_fun = jacobian(self.f1_fun)
        self.j2_fun = jacobian(self.f2_fun)

        self.n_experiments = n_experiments
        self.n_global = len(settings.global_parameters)
        self.n_local = int((len(x0) - self.n_global) / n_experiments)
        assert self.n_global > 0, "No global parameters found. Multi-experiment PE does not make sense in this case!"
        self.n_constraints = len(self.f2_fun(np.hstack((x0[:self.n_local], x0[-self.n_global:]))))
        self.n_controls = len([settings.param_h, settings.param_f])
        self.n_total_parameters = self.n_experiments * self.n_local + self.n_global

        self.level_function_history = []
        self.success = True

        self.L = []
        self.Q = []
        self.R = []
        self.T = []
        self.Gtilde = []
        self.G_idx = np.array([])
        self.Rg = np.array([])
        self.Pg = np.array([])
        self.S0 = np.array([])
        self.E0 = np.array([]).reshape(0, self.n_total_parameters)
        self.fS0 = np.array([])
        self.fE0 = np.array([])
        self.T_alpha_inv = np.array([])
        self.P_inv = np.array([])
        self.T_beta_inv = np.array([])
        self.ci = np.zeros_like(x0)

        self.x = copy.deepcopy(x0)  # initial guess
        self.alpha = 1

        for i in range(MAX_ITERS):
            # construct multi-experiment function evaluation
            self.f = np.array([])
            self.construct_f(x=self.x)
            # construct multi-experiment jacobian
            self.J = np.array([]).reshape(0, self.n_total_parameters)
            self.construct_jacobian(x=self.x)
            if not check_PD(j=self.J):
                logging.warning(f"Positive definiteness does not hold in iterate {i}!")
            dxbar = self.solve_linearized_system_for_step_size()
            self.restrictive_monotonicity_test(x=self.x, dx=dxbar)
            dx = dxbar * self.alpha
            self.alpha = np.minimum(1, ETA / (self.compute_w(x=self.x, dx=dxbar) * np.linalg.norm(dxbar)))
            step = copy.deepcopy(dx)
            if np.linalg.norm(step) < CONVERGENCE_THRESHOLD:
                if compute_ci:
                    self.compute_confidence_intervals_with_breakdown()
                break
            self.x = self.x + dx
            self.level_function_history.append(self.level_function(x=self.x))

            if i == MAX_ITERS - 1:
                self.success = False

        plt.plot(np.arange(0, len(self.level_function_history)), self.level_function_history, '*')
        plt.xlabel('iterations')
        plt.ylabel('level function')
        plt.show()

    def construct_jacobian(self, x):
        self.S0 = self.j1_fun(x)
        y = self.split_into_experiments(x)
        self.E0 = np.array([]).reshape(0, self.n_total_parameters)
        for i in range(self.n_experiments):
            E = np.zeros((self.j2_fun(y[i]).shape[0], self.n_total_parameters))
            E[:, i * self.n_local:(i + 1) * self.n_local] = self.j2_fun(y[i])[:, :self.n_local]
            E[:, self.n_experiments * self.n_local:] = self.j2_fun(y[i])[:, self.n_local:]
            assert check_CQ(E), f"Constraint qualification does not hold in experiment {i}!"
            self.J = np.row_stack((self.J, E, self.S0[i*self.n_controls:(i+1)*self.n_controls]))
            self.E0 = np.row_stack((self.E0, E))
        assert check_CQ(self.E0), "Constraint qualification does not hold in the multi-experiment matrix!"

    def construct_f(self, x):
        self.fS0 = self.f1_fun(x)
        y = self.split_into_experiments(x)
        self.fE0 = np.array([])
        for i in range(self.n_experiments):
            fE = self.f2_fun(y[i])
            self.f = np.concatenate((self.f, fE, self.fS0[i*self.n_controls:(i+1)*self.n_controls]))
            self.fE0 = np.concatenate((self.fE0, fE))

    def split_into_experiments(self, x):
        y = x[:-self.n_global].reshape(self.n_experiments, self.n_local)
        return np.column_stack((y, np.tile(x[-self.n_global:].reshape(-1, 1), self.n_experiments).T))

    def compute_T_alpha(self, J):
        self.L = [np.array([])] * self.n_experiments
        self.P = [np.array([])] * self.n_experiments
        self.Q = [np.array([])] * self.n_experiments
        self.R = [np.array([])] * self.n_experiments
        self.T = [np.array([])] * self.n_experiments
        for i in range(self.n_experiments):
            row_idx = slice(i * (self.n_constraints + self.n_controls), (i+1) * (self.n_constraints + self.n_controls))
            J_local = J[row_idx, i * self.n_local:(i+1) * self.n_local]
            E, S = J_local[:self.n_constraints, :], J_local[self.n_constraints:, :]
            _, _, P = scipy.linalg.qr(E, mode='full', pivoting=True)
            self.P[i] = np.eye(P.shape[0])[P]
            E_tilde, S_tilde = E @ self.P[i].T, S @ self.P[i].T
            T, R11 = np.linalg.qr(E_tilde[:, :self.n_constraints], mode='complete')
            R12 = T.T @ E_tilde[:, -1:]
            self.L[i] = S_tilde[:, :self.n_constraints] @ np.linalg.inv(R11)
            self.Q[i], R31 = np.linalg.qr(S_tilde[:, self.n_constraints:] - self.L[i] @ R12, mode='complete')
            self.R[i] = np.block([[R11, R12], [np.zeros((self.n_controls, self.n_constraints)), R31]])
            self.T[i] = np.block([[T, np.zeros((self.n_constraints, self.n_controls))], [self.L[i], self.Q[i]]])
            assert np.allclose(self.T[i] @ self.R[i], J_local @ self.P[i].T), "Jacobian decomposition is inconsistent!"
        return scipy.linalg.block_diag(*self.T)

    def solve_linearized_system_for_step_size(self):
        # step 1
        T_alpha = self.compute_T_alpha(self.J)
        self.T_alpha_inv = np.linalg.inv(T_alpha)
        Pt = scipy.linalg.block_diag(scipy.linalg.block_diag(*self.P).T, np.eye(self.n_global))
        J1 = self.T_alpha_inv @ self.J @ Pt
        n_rows = self.n_constraints + self.n_controls
        self.G_idx = np.concatenate([np.where(~self.R[i].any(axis=1))[0] + i*n_rows for i in range(self.n_experiments)])
        Ji_idx = np.setdiff1d(np.arange(J1.shape[0]), self.G_idx)
        self.P_inv = np.eye(J1.shape[0])[np.concatenate((Ji_idx, self.G_idx))]
        J2 = self.P_inv @ J1
        # step 3
        assert len(self.G_idx)
        _, _, Pgidx = scipy.linalg.qr(J2[-len(self.G_idx):, -self.n_global:], mode='full', pivoting=True)
        self.Pg = np.eye(Pgidx.shape[0])[Pgidx]
        T_beta_g, Rg = np.linalg.qr(J2[-len(self.G_idx):, -self.n_global:] @ self.Pg.T, mode='complete')
        T_beta = scipy.linalg.block_diag(np.eye(len(Ji_idx)), T_beta_g)
        self.T_beta_inv = np.linalg.inv(T_beta)
        J = self.T_beta_inv @ J2
        f = self.T_beta_inv @ self.P_inv @ self.T_alpha_inv @ self.f
        assert np.allclose(T_alpha @ self.P_inv.T @ T_beta @ J, self.J @ Pt), "Jacobian decomposition is inconsistent!"
        # compute step size for the global variables
        self.Rg = Rg[:Rg.shape[1], :]
        fG = f[-len(self.G_idx):][:self.Rg.shape[1]]
        dxG = -np.linalg.inv(self.Rg @ self.Pg) @ fG
        # compute the step size for the local variables
        dx = np.zeros((self.n_experiments, self.n_local))
        init = 0
        self.Gtilde = [np.array([])] * self.n_experiments
        for i in range(self.n_experiments):
            i_idx = np.where(~np.isclose(np.sum(self.R[i], axis=1), 0))[0]
            f_idx = slice(init, init+len(i_idx))
            self.Gtilde[i] = J[f_idx, -self.n_global:]
            dx[i] = -np.linalg.inv((self.R[i] @ self.P[i])[i_idx]) @ (self.Gtilde[i] @ dxG + f[f_idx])
            init += len(i_idx)
        dxbar = np.concatenate((dx.flatten(), dxG))
        assert np.isclose(max(np.abs(self.E0 @ dxbar + self.fE0)), 0)
        return dxbar

    def level_function(self, x):
        f1 = 0.5 * np.linalg.norm(self.f1_fun(x)) ** 2
        y = self.split_into_experiments(x)
        f2 = np.concatenate([self.f2_fun(y[i]) for i in range(self.n_experiments)])
        f2 = np.sum(np.abs(f2))
        return f1+f2

    def compute_w(self, x, dx):
        f = np.array([])
        fS0 = self.f1_fun(x + self.alpha * dx)
        y = self.split_into_experiments(x + self.alpha * dx)
        fE0 = np.array([])
        for i in range(self.n_experiments):
            fE = self.f2_fun(y[i])
            f = np.concatenate((f, fE, fS0[i * self.n_controls:(i + 1) * self.n_controls]))
            fE0 = np.concatenate((fE0, fE))
        f_tilde = self.T_beta_inv @ self.P_inv @ self.T_alpha_inv @ f
        fG = f_tilde[-len(self.G_idx):][:self.Rg.shape[1]]
        dxG = -np.linalg.inv(self.Rg @ self.Pg) @ fG
        dxbar = np.zeros((self.n_experiments, self.n_local))
        init = 0
        for i in range(self.n_experiments):
            i_idx = np.where(~np.isclose(np.sum(self.R[i], axis=1), 0))[0]
            f_idx = slice(init, init + len(i_idx))
            dxbar[i] = -np.linalg.inv((self.R[i] @ self.P[i])[i_idx]) @ (self.Gtilde[i] @ dxG + f_tilde[f_idx])
            init += len(i_idx)
        dxbar = np.concatenate((dxbar.flatten(), dxG))
        return 2 * np.linalg.norm(dxbar - (1 - self.alpha) * dx) / ((self.alpha * np.linalg.norm(dx)) ** 2)

    def restrictive_monotonicity_test(self, x, dx):
        w = self.compute_w(x=x, dx=dx)
        while self.alpha * w * np.linalg.norm(dx) > ETA_MAX:
            self.alpha = ETA / (w * np.linalg.norm(dx))
            w = self.compute_w(x=x, dx=dx)

    def compute_confidence_intervals_with_breakdown(self):
        Ug = np.dot(np.linalg.inv(self.Rg), np.linalg.inv(self.Rg).T)
        beta = np.linalg.norm(self.f1_fun(self.x)) / np.sqrt(self.J.shape[0] - self.J.shape[1])
        self.ci[-self.n_global:] = beta * np.sqrt(np.diag(self.Pg.T @ Ug @ self.Pg))
        for i in range(self.n_experiments):
            R = self.R[i][:self.n_local, :self.n_local]
            R1 = self.R[i][:self.n_constraints, :self.n_constraints]
            R2 = self.R[i][:self.n_constraints, self.n_constraints:]
            R3 = self.R[i][self.n_constraints:self.n_local, self.n_constraints:]
            D = np.dot(np.linalg.inv(R1), np.linalg.inv(R1).T)
            E = -np.dot(np.linalg.inv(R1), R2)
            F = np.dot(np.linalg.inv(R3), np.linalg.inv(R3).T)
            A = np.dot(np.linalg.inv(R), self.Gtilde[i])
            U = np.block([[D + E @ F @ E.T, E @ F], [F @ E.T, F]])
            Ci = np.diag(self.P[i].T @ (U + A @ Ug @ A.T) @ self.P[i])
            self.ci[i * self.n_local:(i + 1) * self.n_local] = beta * np.sqrt(Ci)

    def compute_confidence_intervals_without_breakdown(self):
        Ug = np.dot(np.linalg.inv(self.Rg), np.linalg.inv(self.Rg).T)
        beta = np.linalg.norm(self.f1_fun(self.x)) / np.sqrt(self.J.shape[0] - self.J.shape[1])
        self.ci[-self.n_global:] = beta * np.sqrt(np.diag(self.Pg.T @ Ug @ self.Pg))
        for i in range(self.n_experiments):
            R = self.R[i][:self.n_local, :self.n_local]
            res = self.Gtilde[i] @ Ug @ self.Gtilde[i].T + np.eye(R.shape[0])
            Ci = np.diag(self.P[i].T @ np.linalg.inv(R) @ res @ np.linalg.inv(R).T @ self.P[i])
            self.ci[i * self.n_local:(i + 1) * self.n_local] = beta * np.sqrt(Ci)
