import autograd.numpy as np
import matplotlib.pyplot as plt
from ..utils.timer import timing_decorator
from ..models.utils import nparray_to_dict
from ..parameter_estimation.problem_generator import OptimizationProblemGenerator
from ..optimization.multi_experiment.select_optimizer import import_optimizer
from ..postprocessing.plot_decorator import handle_plots
from ..logging_ import logger


class ParameterEstimator:
    def __init__(
        self,
        x0: np.ndarray,
        model: object,
        method: str = "osqp",
        xtol: float = 1e-4,
        max_iters: int = 100,
        plot_iters: bool = False,
        compute_ci: bool = False,
        timer: bool = False,
    ):
        """
        Solve the multi-experiment parameter estimation problem.

        Parameters
        ----------
        x0 : np.ndarray
            initial guess
        model : object
            details of the model
        method : str
            approach to solve the problem
        xtol : float
            convergence threshold
        max_iters : int
            maximum number of iterations
        plot_iters : bool
            whether to plot the level function at each iteration
        compute_ci : bool
            whether to compute confidence intervals
        timer : bool
            whether to time the solver
        """
        if x0 is None:
            return None

        self.x0 = x0
        self.model = model
        self.xtol = xtol
        self.max_iters = max_iters
        self.plot_iters = plot_iters
        self.compute_ci = compute_ci
        self.method = method

        self.n_experiments = len(model.data)
        self.n_observables = len(model.controls)
        self.n_global = len(model.global_parameters)
        self.n_local = len(model.compartments) + len(model.controls)
        if model.bifurcation_type == "saddle-node":
            self.n_local += len(model.compartments)
        elif model.bifurcation_type == "hopf":
            self.n_local += 2 * len(model.compartments) + 1
        else:
            raise Exception("Invalid bifurcation type!")

        self.problem = OptimizationProblemGenerator(
            model=model,
            include_steady_state=True,
            include_singularity=True,
            include_normalization=True,
        )
        self.equality_constraints = self.problem.stack_functions

        self.Solver = import_optimizer(method)

        if timer:
            self.__run_solver = timing_decorator(self.__run_solver)
        logger.info(f"Estimate the model parameters using {method} solver.")
        self.result = self.__run_solver()

        if self.result.success:
            logger.info(f"Solver has converged in {self.result.n_iters} iterations!")
            logger.info(f"Initial guesses: {self.__get_global_parameters(self.x0)}.")
            logger.info(f"Solutions: {self.__get_global_parameters(self.result.x)}.")
            if compute_ci:
                CI = self.__get_global_parameters(self.result.confidence_intervals)
                logger.info(f"Confidence intervals: {CI}.")
            self.__plot_results()

    def __run_solver(self):
        self.solver = self.Solver(
            x0=self.x0,
            f1_fun=self.objective_function,
            f2_fun=self.equality_constraints,
            n_local=self.n_local,
            n_global=self.n_global,
            n_observables=self.n_observables,
            n_experiments=self.n_experiments,
            xtol=self.xtol,
            max_iters=self.max_iters,
            plot_iters=self.plot_iters,
            compute_ci=self.compute_ci,
        )
        self.solver.minimize()
        return self.solver.result

    def __get_global_parameters(self, x: np.ndarray) -> dict:
        """
        Get the global parameters from the multi-experiment solution vector.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        dict : global parameters
        """
        parameters = dict()
        for i, key in enumerate(self.model.global_parameters):
            parameters[key] = x[-self.n_global + i]
        return parameters

    @handle_plots(plot_name="fitting_results")
    def __plot_results(self):
        """
        Plot the results of the parameter estimation.
        """
        h_param = self.model.controls["homotopy"]
        f_param = self.model.controls["free"]

        h_data = np.array([d[h_param] for d in self.model.data])
        f_data = np.array([d[f_param] for d in self.model.data])

        solutions = self.solver.split_into_experiments(self.result.x)
        initial_guesses = self.solver.split_into_experiments(self.x0)

        fig, ax = plt.subplots()
        ax.plot(h_data, f_data, "X", color="black")
        for i in range(self.n_experiments):
            _, p0, _ = nparray_to_dict(initial_guesses[i], model=self.model)
            ax.plot(p0[h_param], p0[f_param], "o", color="red", alpha=0.2)

            _, p, _ = nparray_to_dict(solutions[i], model=self.model)
            if self.compute_ci:
                CI = self.result.confidence_intervals
                error = self.solver.split_into_experiments(CI)
                _, perr, _ = nparray_to_dict(error[i], model=self.model)
                _, _, bars = ax.errorbar(
                    x=p[h_param],
                    y=p[f_param],
                    xerr=perr[h_param],
                    yerr=perr[f_param],
                    ecolor="k",
                    marker="o",
                    mfc="r",
                )
                [bar.set_alpha(0.4) for bar in bars]
            else:
                ax.plot(p[h_param], p[f_param], "o", color="red")

        ax.set_xlabel(h_param)
        ax.set_ylabel(f_param)
        return _, fig

    def objective_function(self, x: np.ndarray) -> np.ndarray:
        """
        Objective function for the parameter estimation problem.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : residuals
        """
        obj_fun = np.array([])
        global_x = x[-self.n_global :]
        for i, data in enumerate(self.model.data):
            local_x = x[i * self.n_local : (i + 1) * self.n_local]
            solution = np.concatenate((local_x, global_x))
            _, p, _ = nparray_to_dict(solution, model=self.model)

            for key in self.model.controls.values():
                if self.model.measurement_error == "relative_linear":
                    residual = (p[key] - data[key]) / data[key]
                elif self.model.measurement_error == "relative_log":
                    residual = (np.log10(p[key]) - np.log10(data[key])) / np.log10(data[key])
                elif self.model.measurement_error == "absolute_linear":
                    residual = p[key] - data[key]
                elif self.model.measurement_error == "absolute_log":
                    residual = np.log10(p[key]) - np.log10(data[key])
                else:
                    raise Exception("Invalid measurement error type!")

                obj_fun = np.hstack((obj_fun, residual))

        return obj_fun
