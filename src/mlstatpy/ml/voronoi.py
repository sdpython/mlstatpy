# -*- coding: utf-8 -*-
"""
@file
@brief About Voronoi Diagram
"""
import numpy
import warnings
from sklearn.linear_model import LinearRegression
from mlinsights.mlmodel import QuantileLinearRegression
from statsmodels.regression.quantile_regression import QuantReg


def voronoi_estimation(L, B, C=None, D=None, cl=0, qr=True):
    """
    Determines a Voronoi diagram close to a convex
    partition defined by a logistic regression in *n* classes.
    :math:`M \\in \\mathbb{M}_{nd}` a row matrix :math:`(L_1, ..., L_n)`.
    Every border between two classes *i* and *j* is defined by:
    :math:`\\scal{L_i}{X} + B = \\scal{L_j}{X} + B`.

    The function looks for a set of points from which the Voronoi
    diagram can be inferred. It is done through a linear regression
    with norm *L1*. See :ref:`l-lrvor-connection`.

    @param      L       matrix
    @param      B       vector
    @param      C       additional conditions (see below)
    @param      D       addition condictions (see below)
    @param      cl      class on which the additional conditions applies
    @param      qr      use quantile regression
    @return             matrix :math:`P \\in \\mathbb{M}_{nd}`

    The function solves the linear system:

    .. math::

        \\begin{array}{rcl}
        & \\Longrightarrow & \\left\\{\\begin{array}{l}\\scal{\\frac{L_i-L_j}{\\norm{L_i-L_j}}}{P_i + P_j} + 2 \\frac{B_i - B_j}{\\norm{L_i-L_j}} = 0 \\\\
        \\scal{P_i-  P_j}{u_{ij}} - \\scal{P_i - P_j}{\\frac{L_i-L_j}{\\norm{L_i-L_j}}} \\scal{\\frac{L_i-L_j}{\\norm{L_i-L_j}}}{u_{ij}}=0
        \\end{array} \\right.
        \\end{array}

    If the number of dimension is big and
    the number of classes small, the system has
    multiple solution. Addition condition must be added
    such as :math:`CP_i=D` where *i=cl*, :math:`P_i`
    is the Voronoï point attached to class *cl*.
    `Quantile regression <https://fr.wikipedia.org/wiki/R%C3%A9gression_quantile>`_
    is not implemented in :epkg:`scikit-learn`.
    We use `QuantReq <http://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html>`_.
    """
    matL = []
    matB = []
    for i in range(0, L.shape[0]):
        for j in range(i + 1, L.shape[0]):
            li = L[i, :]
            lj = L[j, :]
            c = (li - lj)
            nc = (c.T @ c) ** 0.5

            # first condition
            mat = numpy.zeros((L.shape))
            mat[i, :] = c
            mat[j, :] = c
            d = -2 * (B[i] - B[j])
            matB.append(d)
            matL.append(mat.ravel())

            # condition 2 - hides multiple equation
            # we pick one
            coor = 0
            found = False
            while not found and coor < len(c):
                if c[coor] == 0:
                    coor += 1
                    continue
                if c[coor] == nc:
                    coor += 1
                    continue
                found = True
            if not found:
                raise ValueError(
                    "Matrix L has two similar rows {0} and {1}. Problem cannot be solved.".format(i, j))

            c /= nc
            c2 = c * c[coor]
            mat = numpy.zeros((L.shape))
            mat[i, :] = -c2
            mat[j, :] = c2

            mat[i, coor] += 1
            mat[j, coor] -= 1
            matB.append(0)
            matL.append(mat.ravel())

    nbeq = (L.shape[0] * (L.shape[0] - 1)) // 2
    matL = numpy.array(matL)
    matB = numpy.array(matB)
    if nbeq * 2 <= L.shape[0] * L.shape[1]:
        if C is None and D is None:
            warnings.warn(
                "[voronoi_estimation] Additional condition are required.")
        if C is not None and D is not None:
            matL = numpy.vstack([matL, numpy.zeros((1, matL.shape[1]))])
            a = cl * L.shape[1]
            b = a + L.shape[1]
            matL[-1, a:b] = C
            if not isinstance(D, float):
                raise TypeError("D must be a float not {0}".format(type(D)))
            matB = numpy.hstack([matB, [D]])
        elif C is None and D is None:
            pass
        else:
            raise ValueError(
                "C and D must be None together or not None together.")

    if qr:
        clq = QuantileLinearRegression(
            fit_intercept=False, max_iter=max(matL.shape))
        clq.fit(matL, matB)
        res = clq.coef_
    else:
        clr = LinearRegression(fit_intercept=False)
        clr.fit(matL, matB)
        res = clr.coef_
    res = res.reshape(L.shape)
    return res
