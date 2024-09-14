from miniature_octo_chainsaw.logging_ import logger
from miniature_octo_chainsaw.select.mathematical_model import import_model
from miniature_octo_chainsaw.select.data_file import import_data
from miniature_octo_chainsaw.data_organizer import add_noise_to_data, filter_data
from miniature_octo_chainsaw.parameter_estimation.initial_guess import InitialGuessGenerator
from miniature_octo_chainsaw.parameter_estimation.problem import ParameterEstimation
from miniature_octo_chainsaw.parameter_estimation.results import save_results_as_pickle


def main(MODEL_NAME):

    logger.setLevel("INFO")

    # Load the model and data
    model = import_model(name=MODEL_NAME)
    data = import_data(name=MODEL_NAME)

    model.generate_parameter_guesses()
    logger.info(f"True model parameters: {model.true_parameters}")
    logger.info(f"Parameter guess initialization: {model.parameters}")

    # Preprocess the data
    data = filter_data(data=data, desired_length=25)
    data = add_noise_to_data(data=data, noise_scale=model.data_noise)
    model.data = data
    logger.info(f"Loaded {len(data)} data points with {model.data_noise * 100}% noise.")

    # Generate initial guesses for the parameter estimation
    igGenerator = InitialGuessGenerator(model=model)

    # Solve parameter estimation problem
    PE = ParameterEstimation(
        x0=igGenerator.initial_guess,
        mask=igGenerator.mask,
        model=model,
        n_experiments=int(sum(igGenerator.mask)),
        method="gauss-newton",
        plot_iters=True,
        compute_ci=False,
        timer=False,
    )

    save_results_as_pickle(res=PE)


if __name__ == "__main__":
    main("steinmetz_larter")
