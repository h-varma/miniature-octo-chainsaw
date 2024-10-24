import autograd.numpy as np
from ..initial_guess.steady_state import find_steady_state
from ..initial_guess.bifurcation_point import find_bifurcation_point
from ..initial_guess.trace_data import trace_measured_bifurcations
from ..initial_guess.match_solutions import match_solutions_to_data
from ..models.utils import nparray_to_dict
from ..logging_ import logger


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

        try:
            # Step 1: Get a steady state solution
            logger.info("Step 1: Find a steady state solution.")
            self.steady_state, _ = find_steady_state(model=self.model)

            # Step 2: Draw a bifurcation diagram and select a bifurcation point for continuation
            logger.info("Step 2: Continue the steady state to draw a bifurcation diagram.")
            kwargs = {"model": self.model, "continuer_name": "deflated", "optimizer_name": "scipy"}
            self.bifurcation_point, _ = find_bifurcation_point(x0=self.steady_state, **kwargs)

            # Step 3: Continue the bifurcation point to trace the data
            logger.info("Step 3: Trace a two-parameter bifurcation diagram along the data.")
            kwargs = {"model": self.model, "continuer_name": "pseudo-arclength"}
            self.bifurcation_points, _ = trace_measured_bifurcations(
                x0=self.bifurcation_point, **kwargs
            )

            # Step 4: Set up the initial guesses
            logger.info("Step 5: Match the predicted points to experimental data.")
            self.initial_guesses = match_solutions_to_data(
                model=self.model, solutions=self.bifurcation_points
            )

            # Step 6: Append global parameters to the initial guess
            logger.info("Step 6: Append global parameters to the initial guess.")
            for global_parameter in self.model.global_parameters:
                self.model.parameters[global_parameter]["vary"] = True
                value = self.model.parameters[global_parameter]["value"]
                self.initial_guesses = np.hstack([self.initial_guesses, value])
            self.model.mask["global_parameters"] = True
        except Exception as e:
            self.initial_guesses = None
            logger.error(f"Initial guess generation failed: {e}")
