import autograd.numpy as np
from miniature_octo_chainsaw.initial_guess.steady_state import find_steady_state
from miniature_octo_chainsaw.initial_guess.bifurcation_point import find_bifurcation_point
from miniature_octo_chainsaw.initial_guess.trace_data import trace_measured_bifurcations
from miniature_octo_chainsaw.initial_guess.match_solutions import match_solutions_to_data
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
        # Step 1: Get a steady state solution
        logger.info("Step 1: Find a steady state solution.")
        steady_state = find_steady_state(model=model,optimizer_name="scipy")

        # Step 2: Draw a bifurcation diagram and select a bifurcation point for continuation
        logger.info("Step 2: Continue the steady state to draw a bifurcation diagram.")
        kwargs = {"model": model, "continuer_name": "deflated", "optimizer_name": "scipy"}
        bifurcation_point = find_bifurcation_point(x0=steady_state, **kwargs)

        # Step 3: Continue the bifurcation point to trace the data
        logger.info("Step 3: Trace a two-parameter bifurcation diagram along the data.")
        kwargs = {"model": model, "method": "pseudo-arclength"}
        bifurcation_points = trace_measured_bifurcations(x0=bifurcation_point, **kwargs)

        # Step 4: Set up the initial guesses
        logger.info("Step 5: Match the predicted points to experimental data.")
        initial_guesses = match_solutions_to_data(model=model, solutions=bifurcation_points)

        # Step 6: Append global parameters to the initial guess
        logger.info("Step 6: Append global parameters to the initial guess.")
        for global_parameter in model.global_parameters:
            model.parameters[global_parameter]["vary"] = True
            value = model.parameters[global_parameter]["value"]
            initial_guesses = np.hstack([initial_guesses, value])
        model.mask["global_parameters"] = True
