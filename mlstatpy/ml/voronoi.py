import warnings
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_distances
from mlinsights.mlmodel import QuantileLinearRegression


class VoronoiEstimationError(Exception):
    """
    Raised when the algorithm failed.
    """


def voronoi_estimation_from_lr(
    L, B, C=None, D=None, cl=0, qr=True, max_iter=None, verbose=False
):
    """
    Determines a Voronoi diagram close to a convex
    partition defined by a logistic regression in *n* classes.
    :math:`M \\in \\mathbb{M}_{nd}` a row matrix :math:`(L_1, ..., L_n)`.
    Every border between two classes *i* and *j* is defined by:
    :math:`\\scal{L_i}{X} + B = \\scal{L_j}{X} + B`.

    The function looks for a set of points from which the Voronoi
    diagram can be inferred. It is done through a linear regression
    with norm *L1*. See :ref:`l-lrvor-connection`.

    @param      L           matrix
    @param      B           vector
    @param      C           additional conditions (see below)
    @param      D           addition condictions (see below)
    @param      cl          class on which the additional conditions applies
    @param      qr          use quantile regression
    @param      max_iter    number of condition to remove until convergence
    @param      verbose     display information while training
    @return                 matrix :math:`P \\in \\mathbb{M}_{nd}`

    The function solves the linear system:

    .. math::

        \\begin{array}{rcl}
        & \\Longrightarrow & \\left\\{\\begin{array}{l}
        \\scal{\\frac{L_i-L_j}{\\norm{L_i-L_j}}}{P_i + P_j} +
        2 \\frac{B_i - B_j}{\\norm{L_i-L_j}} = 0 \\\\
        \\scal{P_i-  P_j}{u_{ij}} - \\scal{P_i - P_j}{\\frac{L_i-L_j}
        {\\norm{L_i-L_j}}} \\scal{\\frac{L_i-L_j}{\\norm{L_i-L_j}}}{u_{ij}}=0
        \\end{array} \\right.
        \\end{array}

    If the number of dimension is big and
    the number of classes small, the system has
    multiple solution. Addition condition must be added
    such as :math:`CP_i=D` where *i=cl*, :math:`P_i`
    is the Voronoï point attached to class *cl*.
    `Quantile regression <https://fr.wikipedia.org/wiki/R%C3%A9gression_quantile>`_
    is not implemented in :epkg:`scikit-learn`.
    We use `QuantileLinearRegression <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/quantile_regression.html>`_.

    After the first iteration, the function determines
    the furthest pair of points and removes it from the list
    of equations. If *max_iter* is None, the system goes until
    the number of equations is equal to the number of points * 2,
    otherwise it stops after *max_iter* removals. This is not the
    optimal pair to remove as they could still be neighbors but it
    should be a good heuristic.
    """
    labels_inv = {}
    nb_constraints = numpy.zeros((L.shape[0],))
    matL = []
    matB = []
    for i in range(L.shape[0]):
        for j in range(i + 1, L.shape[0]):
            li = L[i, :]
            lj = L[j, :]
            c = li - lj
            nc = (c.T @ c) ** 0.5

            # first condition
            mat = numpy.zeros((L.shape))
            mat[i, :] = c
            mat[j, :] = c
            d = -2 * (B[i] - B[j])
            matB.append(d)
            matL.append(mat.ravel())
            labels_inv[i, j, "eq1"] = len(matL) - 1
            nb_constraints[i] += 1
            nb_constraints[j] += 1

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
                    "Matrix L has two similar rows {0} and {1}. "
                    "Problem cannot be solved.".format(i, j)
                )

            c /= nc
            c2 = c * c[coor]
            mat = numpy.zeros((L.shape))
            mat[i, :] = -c2
            mat[j, :] = c2

            mat[i, coor] += 1
            mat[j, coor] -= 1
            matB.append(0)
            matL.append(mat.ravel())
            labels_inv[i, j, "eq2"] = len(matL) - 1
            nb_constraints[i] += 1
            nb_constraints[j] += 1

    nbeq = (L.shape[0] * (L.shape[0] - 1)) // 2
    matL = numpy.array(matL)
    matB = numpy.array(matB)

    if max_iter is None:
        max_iter = matL.shape[0] - matL.shape[1]

    if nbeq * 2 <= L.shape[0] * L.shape[1]:
        if C is None and D is None:
            warnings.warn(
                "[voronoi_estimation_from_lr] Additional condition are required.",
                stacklevel=0,
            )
        if C is not None and D is not None:
            matL = numpy.vstack([matL, numpy.zeros((1, matL.shape[1]))])
            a = cl * L.shape[1]
            b = a + L.shape[1]
            matL[-1, a:b] = C
            if not isinstance(D, float):
                raise TypeError(f"D must be a float not {type(D)}")
            matB = numpy.hstack([matB, [D]])
        elif C is None and D is None:
            pass
        else:
            raise ValueError("C and D must be None together or not None together.")

    sample_weight = numpy.ones((matL.shape[0],))
    tol = numpy.abs(matL.ravel()).max() * 1e-8 / matL.shape[0]
    order_removed = []
    removed = set()
    for it in range(max(max_iter, 1)):
        if qr:
            clr = QuantileLinearRegression(
                fit_intercept=False, max_iter=max(matL.shape)
            )
        else:
            clr = LinearRegression(fit_intercept=False)

        clr.fit(matL, matB, sample_weight=sample_weight)
        score = clr.score(matL, matB, sample_weight)

        res = clr.coef_
        res = res.reshape(L.shape)

        # early stopping
        if score < tol:
            if verbose:
                print(
                    "[voronoi_estimation_from_lr] iter={0}/{1} "
                    "score={2} tol={3}".format(it + 1, max_iter, score, tol)
                )
            break

        # defines the best pair of points to remove
        dist2 = pairwise_distances(res, res)
        dist = [
            (d, n // dist2.shape[0], n % dist2.shape[1])
            for n, d in enumerate(dist2.ravel())
        ]
        dist = [_ for _ in dist if _[1] < _[2]]
        dist.sort(reverse=True)

        # test equal points
        if dist[-1][0] < tol:
            _, i, j = dist[-1]
            eq1 = labels_inv[i, j, "eq1"]
            eq2 = labels_inv[i, j, "eq2"]
            if sample_weight[eq1] == 0 and sample_weight[eq2] == 0:
                sample_weight[eq1] = 1
                sample_weight[eq2] = 1
                nb_constraints[i] += 1
                nb_constraints[j] += 1
            else:
                keep = (i, j)
                pos = len(order_removed) - 1
                while pos >= 0:
                    i, j = order_removed[pos]
                    if i in keep or j in keep:
                        eq1 = labels_inv[i, j, "eq1"]
                        eq2 = labels_inv[i, j, "eq2"]
                        if sample_weight[eq1] == 0 and sample_weight[eq2] == 0:
                            sample_weight[eq1] = 1
                            sample_weight[eq2] = 1
                            nb_constraints[i] += 1
                            nb_constraints[j] += 1
                            break
                    pos -= 1
                if pos < 0:
                    raise VoronoiEstimationError(
                        "Two classes have been merged in a single Voronoi point "
                        "(dist={0} < {1}). max_iter should be lower than "
                        "{2}".format(dist[-1][0], tol, it)
                    )

        dmax, i, j = dist[0]
        pos = 0
        while (i, j) in removed or nb_constraints[i] == 0 or nb_constraints[j] == 0:
            pos += 1
            if pos == len(dist):
                break
            dmax, i, j = dist[pos]
        if pos == len(dist):
            break
        removed.add((i, j))
        order_removed.append((i, j))
        eq1 = labels_inv[i, j, "eq1"]
        eq2 = labels_inv[i, j, "eq2"]
        sample_weight[eq1] = 0
        sample_weight[eq2] = 0
        nb_constraints[i] -= 1
        nb_constraints[j] -= 1

        if verbose:
            print(
                "[voronoi_estimation_from_lr] iter={0}/{1} "
                "score={2:.3g} tol={3:.3g} del P{4},{5} d={6:.3g}".format(
                    it + 1, max_iter, score, tol, i, j, dmax
                )
            )

    return res
