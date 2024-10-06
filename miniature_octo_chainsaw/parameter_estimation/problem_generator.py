import autograd.numpy as np
from ..models.utils import nparray_to_dict


class OptimizationProblemGenerator:
    """Define all the subproblems of the parameter estimation problem."""

    def __init__(
        self,
        model: object,
        include_steady_state: bool = False,
        include_singularity: bool = False,
        include_normalization: bool = False,
    ):
        """
        Initialize the class.

        Parameters
        ----------
        model : object
            details of the model
        include_steady_state : bool
            include steady state condition
        include_singularity : bool
            include singularity condition
        include_normalization : bool
            include normalization condition
        """
        self.model = model
        self.rhs_ = model.rhs_
        self.jacobian_ = model.jacobian_
        self.type_ = model.bifurcation_type

        self.include_steady_state = include_steady_state
        self.include_singularity = include_singularity
        self.include_normalization = include_normalization

    def stack_functions(self, x: np.ndarray):
        """
        Stack the given functions.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : stacked vector
        """
        if self.include_steady_state:
            steady_state = self.steady_state_condition(x)
        else:
            steady_state = np.array([])

        if self.include_singularity:
            singularity = self.singularity_condition(x)
        else:
            singularity = np.array([])

        if self.include_normalization:
            normalization = self.normalization_condition(x)
        else:
            normalization = np.array([])

        return np.hstack((steady_state, singularity, normalization))

    def steady_state_condition(self, x):
        """
        Steady state condition of the bifurcation point.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : residuals
        """
        return self.rhs_(x)

    def singularity_condition(self, x):
        """
        Singularity condition for the bifurcation point.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : residuals
        """
        c, p, h = nparray_to_dict(x=x, model=self.model)
        jacobian_ = self.jacobian_(x)
        if self.type_ == "hopf":
            h1 = np.dot(jacobian_, h["v"]) + (h["mu"] * h["w"])
            h2 = np.dot(jacobian_, h["w"]) - (h["mu"] * h["v"])
            h3 = np.dot(h["v"], h["w"])
            return np.hstack((h1, h2, h3))
        elif self.type_ == "saddle-node":
            return np.dot(jacobian_, h["h"])
        else:
            raise ValueError("Unrecognized bifurcation type!")

    def normalization_condition(self, x):
        """
        Normalization condition of the bifurcation point.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : residuals
        """
        c, p, h = nparray_to_dict(x=x, model=self.model)
        if self.type_ == "hopf":
            return np.dot(h["v"], h["v"]) + np.dot(h["w"], h["w"]) - 1
        elif self.type_ == "saddle-node":
            return np.dot(h["h"], h["h"]) - 1
        else:
            raise ValueError("Unrecognized bifurcation type!")
