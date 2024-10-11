import sys
import os

file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../..")))

from model_equations import Model
from miniature_octo_chainsaw.logging_ import logger
from miniature_octo_chainsaw.preprocessing.preprocess_data import DataPreprocessor
from miniature_octo_chainsaw.parameter_estimation.initial_guess import InitialGuessGenerator
from miniature_octo_chainsaw.parameter_estimation.parameter_estimator import ParameterEstimator
from miniature_octo_chainsaw.parameter_estimation.results import save_results_as_pickle


def main():

    logger.setLevel("INFO")

    # Load the model and randomize the parameters
    model = Model()
    model.generate_parameter_guesses(truncated=False)

    # Preprocess the data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_the_data(file_path=file_path)
    data_preprocessor.add_noise_to_the_data(scale=model.data_noise)
    data_preprocessor.select_subset_of_data(length=25)
    model.data = data_preprocessor.data

    # Generate initial guesses for the parameter estimation
    initializer = InitialGuessGenerator(model=model)

    # Solve parameter estimation problem
    fit = ParameterEstimator(
        x0=initializer.initial_guesses,
        model=model,
        method="osqp",
        plot_iters=True,
        compute_ci=False,
        timer=False,
    )

    # save_results_as_pickle(res=fit)


if __name__ == "__main__":
    main()
