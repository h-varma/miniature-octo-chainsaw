def select_local_optimizer(optimizer: str):
    """
    Select the local optimizer.

    Parameters
    ----------
    optimizer : str
        name of the optimizer

    Returns
    -------
    object : optimizer object
    """

    if optimizer == "scipy":
        from miniature_octo_chainsaw.optimization.single_experiment.scipy_optimizer import ScipyOptimizer

        return ScipyOptimizer
    elif optimizer == "gauss-newton":
        from miniature_octo_chainsaw.optimization.single_experiment.gauss_newton_optimizer import GaussNewtonOptimizer

        return GaussNewtonOptimizer
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer}")
