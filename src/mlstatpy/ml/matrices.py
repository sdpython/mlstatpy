# -*- coding: utf-8 -*-
"""
@file
@brief Algorithms about matrices.
"""
import warnings
import numpy
import numpy.linalg
from scipy.linalg.lapack import dtrtri  # pylint: disable=E0611


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
        raise ValueError("mat must be a matrix.")  # pragma: no cover
    if mat.shape[1] < mat.shape[0]:
        raise RuntimeError(  # pragma: no cover
            "The function only works if the number of rows is less "
            "than the number of columns.")
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
    :math:`\\norme{y - X\\beta}`, based on the algorithm
    :ref:`Arbre de décision optimisé pour les régressions linéaires
    <algo_decision_tree_mselin>`.

    @param      X       features
    @param      y       targets
    @param      algo    None to use the standard algorithm
                        :math:`\\beta = (X'X)^{-1} X'y`,
                        `'gram'`, `'qr'`
    @return             beta

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import linear_regression

        X = numpy.array([[1., 2., 3., 4.],
                         [5., 6., 6., 6.],
                         [5., 6., 7., 8.]]).T
        y = numpy.array([0.1, 0.2, 0.19, 0.29])
        beta = linear_regression(X, y, algo="gram")
        print(beta)

    ``algo=None`` computes :math:`\\beta = (X'X)^{-1} X'y`.
    ``algo='qr'`` uses a `QR <https://docs.scipy.org/doc/numpy/reference
    /generated/numpy.linalg.qr.html>`_ decomposition and calls function
    `dtrtri <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
    linalg.lapack.dtrtri.html>`_ to invert an upper triangular matrix.
    ``algo='gram'`` uses :func:`gram_schmidt
    <mlstatpy.ml.matrices.gram_schmidt>` and then computes
    the solution of the linear regression (see above for a link
    to the algorithm).
    """
    if len(y.shape) != 1:
        warnings.warn(  # pragma: no cover
            "This function is not tested for a multidimensional linear regression.")
    if algo is None:
        inv = numpy.linalg.inv(X.T @ X)
        return inv @ (X.T @ y)
    if algo == "gram":
        T, P = gram_schmidt(X.T, change=True)
        # T = P X
        return (y.T @ T.T @ P).ravel()
    if algo == "qr":
        Q, R = numpy.linalg.qr(X, "full")
        Ri = dtrtri(R)[0]
        gamma = (y.T @ Q).ravel()
        return (gamma @ Ri.T).ravel()
    raise ValueError(  # pragma: no cover
        "Unknwown algo='{}'.".format(algo))


def norm2(X):
    """
    Computes the square norm for all rows of a
    matrix.
    """
    res = numpy.empty(X.shape[1])
    for i in range(X.shape[1]):
        res[i] = numpy.dot(X[i], X[i])
    return res


def streaming_gram_schmidt_update(Xk, Pk):
    """
    Updates matrix :math:`P_k` to produce :math:`P_{k+1}`
    which is the matrix *P* in algorithm
    :ref:`Streaming Linear Regression
    <algo_reg_lin_gram_schmidt_streaming>`.
    The function modifies the matrix *Pk*
    given as an input.

    @param      Xk      kth row
    @param      Pk      matrix *P* at iteration *k-1*
    """
    tki = Pk @ Xk
    idi = numpy.identity(Pk.shape[0])

    for i in range(0, Pk.shape[0]):
        val = tki[i]

        if i > 0:
            # for j in range(0, i):
            #    d = tki[j] * val
            #    tki[i] -= tki[j] * d
            #    Pk[i, :] -= Pk[j, :] * d
            #    idi[i, :] -= idi[j, :] * d

            dv = tki[:i] * val
            tki[i] -= numpy.dot(dv, tki[:i])
            dv = dv.reshape((i, 1))
            Pk[i, :] -= numpy.multiply(Pk[:i, :], dv).sum(axis=0)
            idi[i, :] -= numpy.multiply(idi[:i, :], dv).sum(axis=0)

        d = numpy.square(idi[i, :]).sum()  # pylint: disable=E1101
        d = tki[i] ** 2 + d
        if d > 0:
            d **= 0.5
            d = 1. / d
            tki[i] *= d
            Pk[i, :] *= d
            idi[i, :] *= d


def streaming_gram_schmidt(mat, start=None):
    """
    Solves the linear regression problem,
    find :math:`\\beta` which minimizes
    :math:`\\norme{y - X\\beta}`, based on
    algorithm :ref:`Streaming Gram-Schmidt
    <algo_reg_lin_gram_schmidt_streaming>`.

    @param      mat     matrix
    @param      start   first row to start iteration, ``X.shape[1]`` by default
    @return             iterator on

    The function assumes the matrix *mat* is
    horizontal: it has more columns than rows.

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import streaming_gram_schmidt

        X = numpy.array([[1, 0.5, 10., 5., -2.],
                         [0, 0.4, 20, 4., 2.],
                         [0, 0.7, 20, 4., 2.]], dtype=float).T

        for i, p in enumerate(streaming_gram_schmidt(X.T)):
            print("iteration", i, "\\n", p)
            t = X[:i+3] @ p.T
            print(t.T @ t)
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")  # pragma: no cover
    if mat.shape[1] < mat.shape[0]:
        raise RuntimeError("The function only works if the number of rows is less "
                           "than the number of columns.")
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


def streaming_linear_regression_update(Xk, yk, XkXk, bk):
    """
    Updates coefficients :math:`\\beta_k` to produce :math:`\\beta_{k+1}`
    in :ref:`l-piecewise-linear-regression`.
    The function modifies the matrix *Pk*
    given as an input.

    @param      Xk      kth row
    @param      yk      kth target
    @param      XkXk    matrix :math:`X_{1..k}'X_{1..k}', updated by the function
    @param      bk      current coefficient (updated by the function)
    """
    Xk = Xk.reshape((1, XkXk.shape[0]))
    xxk = Xk.T @ Xk
    XkXk += xxk
    err = Xk.T * (yk - Xk @ bk)
    bk[:] += (numpy.linalg.inv(XkXk) @ err).flatten()


def streaming_linear_regression(mat, y, start=None):
    """
    Streaming algorithm to solve a linear regression.
    See :ref:`l-piecewise-linear-regression`.

    @param      mat     features
    @param      y       expected target
    @return             iterator on coefficients

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import streaming_linear_regression, linear_regression

        X  = numpy.array([[1, 0.5, 10., 5., -2.],
                          [0, 0.4, 20, 4., 2.],
                          [0, 0.7, 20, 4., 3.]], dtype=float).T
        y  = numpy.array([1., 0.3, 10, 5.1, -3.])

        for i, bk in enumerate(streaming_linear_regression(X, y)):
            bk0 = linear_regression(X[:i+3], y[:i+3])
            print("iteration", i, bk, bk0)
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")  # pragma: no cover
    if mat.shape[0] < mat.shape[1]:
        raise RuntimeError("The function only works if the number of rows is more "
                           "than the number of columns.")
    if len(y.shape) != 1:
        warnings.warn(  # pragma: no cover
            "This function is not tested for a multidimensional linear regression.")
    if start is None:
        start = mat.shape[1]

    Xk = mat[:start]
    XkXk = Xk.T @ Xk
    bk = numpy.linalg.inv(XkXk) @ (Xk.T @ y[:start])
    yield bk

    k = start
    while k < mat.shape[0]:
        streaming_linear_regression_update(mat[k], y[k], XkXk, bk)
        yield bk
        k += 1


def streaming_linear_regression_gram_schmidt_update(Xk, yk, Xkyk, Pk, bk):
    """
    Updates coefficients :math:`\\beta_k` to produce :math:`\\beta_{k+1}`
    in :ref:`Streaming Linear Regression
    <l-piecewise-linear-regression-gram_schmidt>`.
    The function modifies the matrix *Pk*
    given as an input.

    @param      Xk      kth row
    @param      yk      kth target
    @param      Xkyk    matrix :math:`X_{1..k}' y_{1..k}' (updated by the function)
    @param      Pk      Gram-Schmidt matrix produced by the streaming algorithm
                         (updated by the function)
    @param      bk      current coefficient (updated by the function)
    """
    Xk = Xk.T
    streaming_gram_schmidt_update(Xk, Pk)
    Xkyk += (Xk * yk).reshape(Xkyk.shape)
    bk[:] = Pk @ Xkyk @ Pk


def streaming_linear_regression_gram_schmidt(mat, y, start=None):
    """
    Streaming algorithm to solve a linear regression with
    Gram-Schmidt algorithm.
    See :ref:`l-piecewise-linear-regression-gram_schmidt`.

    @param      mat     features
    @param      y       expected target
    @return             iterator on coefficients

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.matrices import streaming_linear_regression, linear_regression

        X  = numpy.array([[1, 0.5, 10., 5., -2.],
                          [0, 0.4, 20, 4., 2.],
                          [0, 0.7, 20, 4., 3.]], dtype=float).T
        y  = numpy.array([1., 0.3, 10, 5.1, -3.])

        for i, bk in enumerate(streaming_linear_regression(X, y)):
            bk0 = linear_regression(X[:i+3], y[:i+3])
            print("iteration", i, bk, bk0)
    """
    if len(mat.shape) != 2:
        raise ValueError("mat must be a matrix.")  # pragma: no cover
    if mat.shape[0] < mat.shape[1]:
        raise RuntimeError("The function only works if the number of rows is more "
                           "than the number of columns.")
    if len(y.shape) != 1:
        warnings.warn(  # pragma: no cover
            "This function is not tested for a multidimensional linear regression.")
    if start is None:
        start = mat.shape[1]

    Xk = mat[:start]
    xyk = Xk.T @ y[:start]
    _, Pk = gram_schmidt(Xk.T, change=True)
    bk = Pk @ xyk @ Pk
    yield bk

    k = start
    while k < mat.shape[0]:
        streaming_linear_regression_gram_schmidt_update(
            mat[k], y[k], xyk, Pk, bk)
        yield bk
        k += 1
