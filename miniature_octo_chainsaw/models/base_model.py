from abc import abstractmethod, ABC
import autograd.numpy as np
from scipy.stats import truncnorm


class BaseModel(ABC):
    """
    Base class for all models - not functional on its own.
    """

    def __init__(self):
        self.data = None
        self.parameters = None
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

    def _initialize_parameters(self, parameters: dict):
        """
        Initialize the parameter dictionary with `value` and `vary` as keys.

        Parameters
        ----------
        parameters : dict
            model parameters
        """
        self.parameters = dict()
        for key, value in parameters.items():
            self.parameters[key] = {'value': value, 'vary': False}

    def generate_parameter_guesses(self):
        """
        Generates random initial guesses for the parameters and controls to be estimated.
        """

        parameter_list = self.global_parameters + list(self.controls.values())

        for parameter, parameter_value in self.true_parameters.items():
            if parameter in parameter_list:
                loc = parameter_value
                scale = parameter_value * self.parameter_noise
                a = (0 - loc) / scale
                b = (np.inf - loc) / scale
                # compute truncated normal random variable
                # truncated at a and b standard deviations from loc
                parameter_value = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
            self.parameters[parameter] = {"value": parameter_value, "vary": False}

    def __getattr__(self, attr : str):
        """
        Set the attributes from the settings object as attributes of self.

        Parameters
        ----------
        attr : str
            name of attribute

        Returns
        -------
        object : attribute value
        """
        return getattr(self.settings, attr)
