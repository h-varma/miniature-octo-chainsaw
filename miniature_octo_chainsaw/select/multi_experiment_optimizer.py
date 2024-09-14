def import_multi_experiment_optimizer(optimizer: str):
    """
    Return a single experiment optimizer.

    Parameters
    ----------
    optimizer : str
        name of the optimization method

    Returns
    -------
    object : optimizer object
    """
    if optimizer == "osqp":
        from miniature_octo_chainsaw.optimization.multi_experiment.osqp_optimizer import MultiExperimentOSQP

        return MultiExperimentOSQP

    elif optimizer == "gauss-newton":
        from miniature_octo_chainsaw.optimization.multi_experiment.gauss_newton_optimizer import MultiExperimentGaussNewton

        return MultiExperimentGaussNewton

    else:
        raise ValueError(f"Unknown optimizer name: {optimizer}")
