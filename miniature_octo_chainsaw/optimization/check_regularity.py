import autograd.numpy as np


def check_positive_definiteness(J: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if the matrix has full column rank.

    Parameters
    ----------
    J : np.ndarray
        matrix to check
    tol : float
        singular value tolerance

    Returns
    -------
    bool : True if the matrix has full column rank
    """
    return np.linalg.matrix_rank(J, tol=tol) == J.shape[1]


def check_constraint_qualification(J: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if the matrix has full row rank.

    Parameters
    ----------
    J : np.ndarray
        matrix to check
    tol : float
        singular value tolerance

    Returns
    -------
    bool : True if the matrix has full row rank
    """
    if len(J):
        if np.linalg.matrix_rank(J, tol=tol) != J.shape[0] and len(J.shape) > 1:
            raise Exception("CQ failed!")
    return True
