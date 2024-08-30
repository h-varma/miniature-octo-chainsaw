import autograd.numpy as np
from miniature_octo_chainsaw.models.peroxidase_oxidase.settings import ModelSettings
from miniature_octo_chainsaw.models.utils import nparray_to_dict
from miniature_octo_chainsaw.models.base_model import BaseModel


class Model(BaseModel):
    """
    Domijan, Mirela, and Markus Kirkilionis.
    "Bistability and oscillations in chemical reaction networks."
    Journal of Mathematical Biology 59 (2009): 467-501.

    Attributes
    ----------
    settings : dataclass
        model information, parameters and settings

    Methods
    -------
    model_rhs(x):
        defines the RHS of the model.

    model_jacobian(x):
        defines the Jacobian matrix of the model.
    """

    def __init__(self):
        super().__init__()
        self.settings = ModelSettings()
        self._initialize_parameters(parameters=self.true_parameters)

    def rhs_(self, x: np.ndarray) -> np.ndarray:
        """
        RHS of the model

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : RHS of the model
        """

        c, p, _ = nparray_to_dict(x=x, model=self)
        model_equations = {
            "x1": -2 * p["k1"] * (c["x1"] ** 2)
            + p["k2"]
            + p["k3"] * c["x1"]
            - p["k4"] * c["x1"] * c["x2"],
            "x2": p["k5"] - p["k6"] * c["x2"] - p["k4"] * c["x1"] * c["x2"],
        }

        M_list = [model_equations[key] for key in self.compartments]
        return np.array(M_list)

    def jacobian_(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of the model

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : Jacobian matrix of the model
        """

        c, p, _ = nparray_to_dict(x=x, model=self)
        model_jacobian = {
            "x1": np.array(
                [
                    -4 * p["k1"] * c["x1"] + p["k3"] - p["k4"] * c["x2"],
                    -p["k4"] * c["x1"],
                ]
            ),
            "x2": np.array([-p["k4"] * c["x2"], -p["k6"] - p["k4"] * c["x1"]]),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
