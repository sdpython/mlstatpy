# -*- coding: utf-8 -*-
"""
@file
@brief Algorithms about matrices.
"""
import numpy
import numpy.linalg


def gram_schmidt(mat, change=False):
    """
    Applies the `Gram–Schmidt process
    <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`_.
    Due to performance, every row is considered as a vector.

    @param      mat     matrix
    @param      change  returns the matrix to change the basis
    @return             new matrix or (new matrix, change matrix)

    .. note::
        The implementation could be improved
        by directly using :epkg:`BLAS` function.
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")
    res = numpy.empty(mat.shape)
    if change:
        base = numpy.identity(mat.shape[0])
    for i in range(0, mat.shape[0]):
        res[i, :] = mat[i, :]
        for j in range(0, i):
            d = numpy.dot(res[j, :], mat[i, :])
            res[i, :] -= res[j, :] * d
            if change:
                base[i, :] -= base[j, :] * d
        d = numpy.dot(res[i, :], res[i, :])
        if d > 0:
            d **= 0.5
            res[i, :] /= d
            if change:
                base[i, :] /= d
    return (res, base) if change else res


def linear_regression(X, y, algo=None):
    """
    Solves the linear regression problem,
    find :math:`\\beta` which minimizes
    :math:`\\norme{y - X\\beta}`, based on
    the algorithm
    :ref:`Arbre de décision optimisé pour les régressions linéaires
    <algo_decision_tree_mselin>`.

    @param      X       features
    @param      y       targets
    @param      algo    None to use the standard algorithm
                        :math:`\\beta = (X'X)^{-1} X'y`,
                        `'gram'` otherwise
    @return             beta
    """
    if algo is None:
        inv = numpy.linalg.inv(X.T @ X)
        return inv @ (X.T @ y)
    elif algo == "gram":
        U, P = gram_schmidt(X.T, change=True)
        U, P = U.T, P.T
        gamma = y.flatten() @ U
        return (P @ gamma).ravel()
    else:
        raise ValueError("Unknwown algo='{}'.".format(algo))
