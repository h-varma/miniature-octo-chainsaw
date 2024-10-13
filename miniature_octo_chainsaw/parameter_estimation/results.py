def get_results(res: object):
    """
    Get a summary of the parameter estimation results.

    Returns
    -------
    dict : results
    """
    result = dict()

    result["data"] = {}
    result["data"]["values"] = res.model.data
    result["data"]["noise"] = res.model.data_noise
    result["data"]["mask"] = res.mask

    result["bifurcation"] = {}
    result["bifurcation"]["type"] = res.model.bifurcation_type
    result["bifurcation"]["continuation_settings"] = res.model.continuation_settings

    result["model"] = {}
    result["model"]["name"] = res.model.name
    result["model"]["parameters"] = res.model.parameters
    result["model"]["compartments"] = res.model.compartments
    result["model"]["controls"] = res.model.controls
    result["model"]["global_parameters"] = res.model.global_parameters

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
