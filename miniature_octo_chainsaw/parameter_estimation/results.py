def get_results(model: object, res: object = None):
    """
    Get a summary of the parameter estimation results.

    Parameters
    ----------
    model : object
        model object
    res : object
        parameter estimation results

    Returns
    -------
    dict : results
    """
    result = dict()

    result["data"] = {}
    result["data"]["values"] = model.data
    result["data"]["noise"] = model.data_noise

    result["bifurcation"] = {}
    result["bifurcation"]["type"] = model.bifurcation_type
    result["bifurcation"]["continuation_settings"] = model.continuation_settings

    result["model"] = {}
    result["model"]["name"] = model.name
    result["model"]["parameters"] = model.parameters
    result["model"]["compartments"] = model.compartments
    result["model"]["controls"] = model.controls
    result["model"]["global_parameters"] = model.global_parameters

    try:
        result["error_message"] = model.error_message
    except AttributeError:
        result["error_message"] = None

    if res.__dict__ != {}:
        result["PE"] = {}
        result["PE"]["n_experiments"] = res.n_experiments
        result["PE"]["initial_guesses"] = res.x0
        result["PE"]["method"] = res.method
        result["PE"]["xtol"] = res.xtol
        result["PE"]["max_iters"] = res.max_iters
        result["PE"]["residual"] = res.model.measurement_error
        result["PE"]["parameter_noise"] = res.model.parameter_noise
        result["PE"]["initial_model_state"] = res.model.initial_state
        result["PE"]["integration_interval"] = res.model.integration_interval

        result["PE"]["result"] = {}
        result["PE"]["result"]["x"] = res.result.x
        result["PE"]["result"]["success"] = res.result.success
        result["PE"]["result"]["func"] = res.result.func
        result["PE"]["result"]["n_iters"] = res.result.n_iters
        result["PE"]["result"]["level_functions"] = res.result.level_functions
        if res.compute_ci:
            result["PE"]["result"]["covariance_matrix"] = res.result.covariance_matrix
            result["PE"]["result"]["confidence_intervals"] = res.result.confidence_intervals
        else:
            result["PE"]["result"]["covariance_matrix"] = None
            result["PE"]["result"]["confidence_intervals"] = None

    return result
