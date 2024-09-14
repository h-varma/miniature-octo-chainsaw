def import_model(name: str) -> object:
    """
    Imports the model equations, jacobian and settings.

    Parameters
    ----------
    name : str
        name of the model

    Returns
    -------
    Model : class
        model class with rhs and jacobian
    """
    if name == "predator_prey":
        from miniature_octo_chainsaw.models.predator_prey.model import Model

    elif name == "peroxidase_oxidase":
        from miniature_octo_chainsaw.models.peroxidase_oxidase.model import Model

    elif name == "steinmetz_larter":
        from miniature_octo_chainsaw.models.steinmetz_larter.model import Model

    else:
        raise Exception(f"{name} model not found.")

    model = Model()
    return model
