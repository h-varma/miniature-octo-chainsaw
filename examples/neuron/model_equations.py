import os
import autograd.numpy as np
from miniature_octo_chainsaw.parser.yaml_parser import YamlParser
from miniature_octo_chainsaw.models.utils import nparray_to_dict
from miniature_octo_chainsaw.models.base_model import BaseModel


class Model(BaseModel):
    """

    Attributes
    ----------
    specifications : dataclass
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
        file_path = os.path.dirname(__file__)
        parser = YamlParser(file_path=file_path)
        self.specifications = parser.get_problem_specifications()

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
            "x": c["y"] - (c["x"] ** 3) + p["b"] * (c["x"] ** 2) + p["I"] - c["z"],
            "y": 1 - 5 * (c["x"] ** 2) - c["y"],
            "z": p["mu"] * (p["s"] * (c["x"] - p["x_rest"]) - c["z"]),
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
            "x": np.array([-3 * (c["x"] ** 2) + 2 * p["b"] * c["x"], 1, -1]),
            "y": np.array([-10 * c["x"], -1, 0]),
            "z": np.array([p["mu"] * p["s"], 0, -p["mu"]]),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
