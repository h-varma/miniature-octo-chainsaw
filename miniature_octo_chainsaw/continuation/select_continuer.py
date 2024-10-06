def import_continuer(method: str):
    """
    Import a continuation method of choice.

    Parameters
    ----------
    method : str
        name of the continuation method

    Returns
    -------
    object : continuation object
    """
    if method == "deflated":
        from ..continuation.deflated_continuation import DeflatedContinuation

        return DeflatedContinuation

    elif method == "pseudo-arclength":
        from ..continuation.pseudo_arclength import PseudoArclengthContinuation

        return PseudoArclengthContinuation

    else:
        raise ValueError(f"Unknown continuation method: {method}")
