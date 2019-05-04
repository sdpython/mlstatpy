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

    @param      mat         matrix
    @param      change      returns the matrix to change the basis
    @return                 new matrix or (new matrix, change matrix)

    The function assumes the matrix *mat* is
    horizontal: it has more columns than rows.

    .. note::
        The implementation could be improved
        by directly using :epkg:`BLAS` function.

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import gram_schmidt

        X = numpy.array([[1., 2., 3., 4.],
                         [5., 6., 6., 6.],
                         [5., 6., 7., 8.]])
        T, P = gram_schmidt(X, change=True)
        print(T)
        print(P)
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")
    if mat.shape[1] < mat.shape[0]:
        raise RuntimeError("The function only works if the number of rows is less "
                           "than the number of column.")
    if change:
        base = numpy.identity(mat.shape[0])
    # The following code is equivalent to:
    # res = numpy.empty(mat.shape)
    # for i in range(0, mat.shape[0]):
    #     res[i, :] = mat[i, :]
    #     for j in range(0, i):
    #         d = numpy.dot(res[j, :], mat[i, :])
    #         res[i, :] -= res[j, :] * d
    #         if change:
    #             base[i, :] -= base[j, :] * d
    #     d = numpy.dot(res[i, :], res[i, :])
    #     if d > 0:
    #         d **= 0.5
    #         res[i, :] /= d
    #         if change:
    #             base[i, :] /= d
    # But it is faster to write it this way:
    res = numpy.empty(mat.shape)
    for i in range(0, mat.shape[0]):
        res[i, :] = mat[i, :]
        if i > 0:
            d = numpy.dot(res[:i, :], mat[i, :])
            m = numpy.multiply(res[:i, :], d.reshape((len(d), 1)))
            m = numpy.sum(m, axis=0)
            res[i, :] -= m
            if change:
                m = numpy.multiply(base[:i, :], d.reshape((len(d), 1)))
                m = numpy.sum(m, axis=0)
                base[i, :] -= m
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
                        `'gram'`, `'gramT'`, `'qr'`, `'qr2'`
    @return             beta

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import linear_regression

        X = numpy.array([[1., 2., 3., 4.],
                         [5., 6., 6., 6.],
                         [5., 6., 7., 8.]]).T
        y = numpy.array([0.1, 0.2, 0.19, 0.29])
        beta = linear_regression(X, y)
        print(beta)
    """
    if algo is None:
        inv = numpy.linalg.inv(X.T @ X)
        return inv @ (X.T @ y)
    elif algo == "gram":
        T, P = gram_schmidt(X.T, change=True)
        gamma = T @ y
        return (gamma.T @ P).ravel()
    elif algo == "qr":
        Q, R = numpy.linalg.qr(X, "full")
        R = numpy.linalg.inv(R)
        gamma = (y.T @ Q).ravel()
        return (gamma @ R.T).ravel()
    else:
        raise ValueError("Unknwown algo='{}'.".format(algo))


def norm2(X):
    """
    Computes the square norm for all rows of a
    matrix.
    """
    return numpy.array([numpy.dot(X[i], X[i]) for i in range(X.shape[1])])


def streaming_gram_schmidt_update(Xi, Pk):
    """
    Updates matrix :math:`P_k` to produce :math:`P_{k+1}`
    which is the matrix *P* in algorithm
    :ref:`Streaming Linear Regression
    <algo_reg_lin_gram_schmidt_streaming>`.
    The function modifies the matrix *Pk*
    given as an input.

    @param      Xi      ith row
    @param      Pk      matrix *P* at iteration *k*
    """
    tki = Pk @ Xi
    idi = numpy.identity(Pk.shape[0])

    for i in range(0, Pk.shape[0]):
        val = tki[i]
        for j in range(0, i):
            d = tki[j] * val
            tki[i] -= tki[j] * d
            Pk[i, :] -= Pk[j, :] * d
            idi[i, :] -= idi[j, :] * d

        d = numpy.square(idi[i, :]).sum()
        d = tki[i] ** 2 + d
        if d > 0:
            d **= 0.5
            tki[i] /= d
            Pk[i, :] /= d
            idi[i, :] /= d


def streaming_gram_schmidt(mat, start=None):
    """
    Solves the linear regression problem,
    find :math:`\\beta` which minimizes
    :math:`\\norme{y - X\\beta}`, based on
    algorithm :ref:`Streaming Linear Regression
    <algo_reg_lin_gram_schmidt_streaming>`.

    @param      mat     matrix
    @param      start   first row to start iteration, ``X.shape[1]`` by default
    @return             iterator on

    The function assumes the matrix *mat* is
    horizontal: it has more columns than rows.
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")
    if mat.shape[1] < mat.shape[0]:
        raise RuntimeError("The function only works if the number of rows is less "
                           "than the number of column.")
    if start is None:
        start = mat.shape[0]
    mats = mat[:, :start]
    _, Pk = gram_schmidt(mats, change=True)
    yield Pk

    k = start
    while k < mat.shape[1]:
        streaming_gram_schmidt_update(mat[:, k], Pk)
        yield Pk
        k += 1
