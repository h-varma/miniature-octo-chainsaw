import autograd.numpy as np
from abc import ABC
from miniature_octo_chainsaw.optimization.single_experiment.base_optimizer import BaseOptimizer


class BaseMultiExperimentOptimizer(BaseOptimizer, ABC):
    """
    Multi-experiment optimizer interface - not functional on its own.
    """
    def __init__(self):
        """
        Initialize the optimizer.
        """
        super().__init__()

    def split_into_experiments(self, x: np.ndarray) -> np.ndarray:
        """
        Split the solution vector into local experiments with common global parameters.

        Parameters
        ----------
        x : np.ndarray
            solution vector

        Returns
        -------
        np.ndarray : solution matrix with experiments as rows
        """
        local_x = x[: -self.n_global]
        local_x = local_x.reshape(self.n_experiments, self.n_local)

        global_x = x[-self.n_global:]
        global_x = global_x.reshape(-1, 1)
        global_x = np.tile(global_x, self.n_experiments)

        return np.column_stack((local_x, global_x.T))
