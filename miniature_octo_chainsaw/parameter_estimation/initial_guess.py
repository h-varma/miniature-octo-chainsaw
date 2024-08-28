import autograd.numpy as np
from miniature_octo_chainsaw.models.integrate import solve_ivp_
from miniature_octo_chainsaw.models.steady_state import solve_rhs
from miniature_octo_chainsaw.problem.continuation import one_parameter_continuation
from miniature_octo_chainsaw.problem.continuation import two_parameter_continuation
from miniature_octo_chainsaw.problem.bifurcation_point import detect_bifurcation_point
from miniature_octo_chainsaw.problem.bifurcation_point import solve_bifurcation_condition
from miniature_octo_chainsaw.models.utils import nparray_to_dict
from miniature_octo_chainsaw.logging_ import logger


class InitialGuessGenerator:
    def __init__(self, model: object):
        """
        Generate initial guesses for the parameter estimation problem.

        Parameters
        ----------
        model : object
            details of the model
        """

        self.model = model
        self.data = model.data
        self.h_param = model.controls["homotopy"]
        self.f_param = model.controls["free"]

        # Step 1: Get a steady state solution
        logger.info("Step 1: Find a steady state solution.")
        solution = solve_ivp_(model=model, plot=True)
        solution = solve_rhs(x0=solution, model=model, local_optimizer="scipy")

        # Step 2: Continue the steady state to draw a bifurcation diagram
        logger.info("Step 2: Continue the steady state to draw a bifurcation diagram.")
        kwargs = {"model": model, "method": "deflated", "local_optimizer": "scipy"}
        solutions = one_parameter_continuation(x0=solution, **kwargs)

        # Step 3: Select a bifurcation point for continuation
        logger.info("Step 3: Select a bifurcation point for continuation.")
        kwargs = {"model": model, "local_optimizer": "scipy"}
        b_point = detect_bifurcation_point(steady_states=solutions, model=model)
        b_point = solve_bifurcation_condition(x0=b_point, **kwargs)

        # Step 4: Continue the bifurcation point to draw a two-parameter bifurcation diagram
        logger.info("Step 4: Trace a two-parameter bifurcation diagram along the data.")
        kwargs = {"model": model, "method": "pseudo-arclength"}
        b_points = two_parameter_continuation(x0=b_point, **kwargs)

        # Step 5: Set up the initial guesses
        logger.info("Step 5: Match the predicted points to experimental data.")
        self.initial_guess = []
        self.mask = np.ones((len(model.data),))
        self.match_solutions_to_data(solutions=b_points)

        # Step 6: Append global parameters to the initial guess
        logger.info("Step 6: Append global parameters to the initial guess.")
        self.append_global_parameters_to_initial_guess()
        model.mask["global_parameters"] = True

    def match_solutions_to_data(self, solutions: list) -> None:
        """
        Match the solutions of two-parameter continuation to the experimental data.

        Parameters
        ----------
        solutions : list
            solutions from the two-parameter continuation
        """

        h_data = np.array([d[self.h_param] for d in self.data])
        f_data = np.array([d[self.f_param] for d in self.data])

        solutions = sorted(solutions, key=lambda x: self.get_parameter_value(x, "f"))
        f_params = list(map(lambda x: self.get_parameter_value(x, "f"), solutions))
        h_params = list(map(lambda x: self.get_parameter_value(x, "h"), solutions))

        for h_value in np.unique(h_data):
            where_h, count_h = self.match_data(data=h_value, solutions=h_params)
            if count_h == 2:
                matching_f = f_data[np.isclose(h_data, h_value)]
                if len(matching_f) == 2:
                    for idx in where_h:
                        self.initial_guess.append(solutions[idx])
                elif len(f_data[where_h]) == 1:
                    idx = np.argmin(abs(f_params[where_h] - matching_f))
                    self.initial_guess.append(solutions[idx])

            elif count_h == 1:
                where_h = where_h[0]
                if len([h for h in h_data if h == h_value]) == 2:
                    _f_data = f_data[np.isclose(h_data, h_value)]
                    f = max(_f_data, key=lambda x: abs(x - f_params[where_h]))
                    self.mask[np.isclose(h_data, h_value) & np.isclose(f_data, f)] = 0
                self.initial_guess.append(solutions[where_h])

            elif count_h == 0:
                self.mask[np.isclose(h_data, h_value)] = 0

        # sort the initial guesses and experimental data so that they match
        self.initial_guess.sort(key=lambda x: self.get_parameter_value(x, "h"))
        iterable_ = zip(self.model.data, self.mask)
        iterable_ = sorted(iterable_, key=lambda x: x[0][self.h_param])

        self.model.data, self.mask = (list(x) for x in zip(*iterable_))
        self.initial_guess = np.hstack(self.initial_guess)

    def append_global_parameters_to_initial_guess(self) -> None:
        """Append global parameters to the initial guess."""
        for global_parameter in self.model.global_parameters:
            self.model.parameters[global_parameter]["vary"] = True
            value = self.model.parameters[global_parameter]["value"]
            self.initial_guess = np.hstack([self.initial_guess, value])

    def get_parameter_value(self, x: np.ndarray, type_: str) -> float:
        """
        Get the value of homotopy or free parameter from the solution array.

        Parameters
        ----------
        x : np.ndarray
            solution array
        type_ : str
            parameter type

        Returns
        -------
        float : parameter value
        """
        if type_ == "h":
            type_ = "homotopy"
        elif type_ == "f":
            type_ = "free"
        else:
            raise ValueError("Unrecognized parameter type!")

        _, p, _ = nparray_to_dict(x, model=self.model)
        return p[self.model.controls[type_]]

    @staticmethod
    def match_data(data: float, solutions: list):
        """
        Find solutions that match the data

        Parameters
        ----------
        data : float
            data point
        solutions : list
            solutions from the two-parameter continuation

        Returns
        -------
        tuple : matching indices and count of matching solutions
        """

        matching_idx = np.where(np.isclose(solutions, data))[0]
        return matching_idx, len(matching_idx)
