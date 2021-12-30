r"""
    Methods for the calculations of the Lyapunov Characteristic Exponents 
    (LCEs).
"""


import numpy as np


def gram_schmidt(U):
    r"""Calculate an orthogonal basis. It is non-vectorized if the shape is 2-D,
    otherwise if the shape is 3-D, the vectorization is done on the first
    dimension.

    ----------
    Parameters
    ----------
    U: array_like
        U (N, M) contains column vectors {U_1, U_2, ..., U_M} with dimension N.

    -------
    Returns
    -------
    W: array_like
        Orthogonal basis calculated from U
    V: array_like
        Orthonormal basis where all column vectors in W are normalized
    """

    if len(U.shape) == 2:
        return __non_vectorized_gs(U)
    elif len(U.shape) == 3:
        return __vectorized_gs(U)


def __non_vectorized_gs(U):
    r"""Follows the usual definition for a given basis U = {u_j}"""

    # Preallocate
    W = np.zeros(U.shape)
    V = np.zeros(U.shape)
    W[:, 0] = U[:, 0]
    V[:, 0] = W[:, 0] / np.linalg.norm(W[:, 0])

    for p in range(1, W.shape[1]):
        tmp_ = np.zeros(W.shape[0])
        for i in range(p):
            tmp_ += np.matmul(U[:, p], V[:, i]) * V[:, i]

        W[:, p] = U[:, p] - tmp_
        V[:, p] = W[:, p] / np.linalg.norm(W[:, p])

    return W, V


def __vectorized_gs(U):
    """Use Einstein convention to ignore the first (time) axis"""

    N, M, P = U.shape

    # Preallocate
    W = np.zeros(U.shape)
    V = np.zeros(U.shape)
    W[:, :, 0] = U[:, :, 0]
    V[:, :, 0] = np.einsum(
        "i...,i->i...", W[:, :, 0], 1 / np.linalg.norm(W[:, :, 0], axis=1)
    )

    for p in range(1, P):
        tmp_ = np.zeros((N, M))
        for i in range(p):
            UdotV = np.einsum("...i,...i", U[:, :, p], V[:, :, i])
            tmp_ += np.einsum("i,i...->i...", UdotV, V[:, :, i])

        W[:, :, p] = U[:, :, p] - tmp_
        V[:, :, p] = np.einsum(
            "i...,i->i...", W[:, :, p], 1 / np.linalg.norm(W[:, :, p], axis=1)
        )

    return W, V
