import numpy as np

PD_ACCURACY_TOLERANCE = 1e-4
CQ_ACCURACY_TOLERANCE = 1e-8


def check_PD(j):
    return np.linalg.matrix_rank(j, tol=PD_ACCURACY_TOLERANCE) == j.shape[1]


def check_CQ(j):
    if len(j):
        if np.linalg.matrix_rank(j) != j.shape[0] and len(j.shape) > 1:
            raise Exception("CQ failed!")
    return True
