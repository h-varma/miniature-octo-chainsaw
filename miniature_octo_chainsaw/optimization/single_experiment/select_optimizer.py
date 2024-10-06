def import_optimizer(optimizer: str):
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
    if optimizer == "scipy":
        from ...optimization.single_experiment.scipy_optimizer import ScipyOptimizer

        return ScipyOptimizer

    elif optimizer == "gauss-newton":
        from ...optimization.single_experiment.gauss_newton_optimizer import GaussNewtonOptimizer

        return GaussNewtonOptimizer

    else:
        raise ValueError(f"Unknown optimizer name: {optimizer}")
