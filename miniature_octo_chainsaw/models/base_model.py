import autograd.numpy as np
from abc import abstractmethod, ABC
from scipy.stats import truncnorm
from ..logging_ import logger


class BaseModel(ABC):
    """
    Base class for all model - not functional on its own.
    """

    def __init__(self):
        self.data = None
        self.mask = {
            "compartments": False,
            "controls": False,
            "auxiliary_variables": False,
            "global_parameters": False,
        }

    @abstractmethod
    def rhs_(self, x: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the model.

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : RHS of the model
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian_(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of the model equations.

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : Jacobian matrix of the model
        """
        raise NotImplementedError

    def generate_parameter_guesses(self, truncated: bool = True):
        """
        Generates random initial guesses for the variable parameters and controls.

        Parameters
        ----------
        truncated : bool
            whether to generate truncated normal random variables
        """
        i = np.random.randint(0, len(self.data))
        for parameter_name, parameter_value in self.true_parameters.items():
            if parameter_name in self.global_parameters:
                loc = parameter_value
                scale = np.abs(parameter_value * self.parameter_noise)
                if truncated:
                    a = (0 - loc) / scale
                    b = (np.inf - loc) / scale
                    # compute truncated normal random variable
                    # truncated at a and b standard deviations from loc
                    parameter_value = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
                else:
                    parameter_value = np.random.normal(loc=loc, scale=scale)
                parameter_value = float(parameter_value)
            elif parameter_name in list(self.controls.values()):
                parameter_value = self.data[i][parameter_name]
            self.parameters[parameter_name] = {"value": parameter_value, "vary": False}

        logger.info(f"True model parameters: {self.true_parameters}")
        logger.info(f"Parameter guess initialization: {self.parameters}")

    def __getattr__(self, attr: str):
        """
        Set the attributes from the ProblemSpecs object as attributes of self.

        Parameters
        ----------
        attr : str
            name of attribute

        Returns
        -------
        object : attribute value
        """
        return getattr(self.specifications, attr)
