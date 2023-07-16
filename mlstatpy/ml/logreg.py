"""
@file
@brief Helpers on logistic regression.
"""
import numpy
from pandas import DataFrame


def random_set_1d(n, kind):
    """
    Builds a random dataset as describes in example
    :ref:`l-example-logistic-decision`.

    @param      n       number of observations
    @param      kind    2, 3, 4 (see example)
    @return             array 2D
    """
    x = numpy.random.rand(n) * 3 - 1
    if kind == 3:
        y = numpy.empty(x.shape, dtype=numpy.int32)
        y[x < 0] = 0
        y[(x >= 0) & (x <= 1)] = 1
        y[x > 1] = 0
    elif kind == 2:
        y = numpy.empty(x.shape, dtype=numpy.int32)
        y[x < 0] = 0
        y[x >= 0] = 1
    elif kind == 4:
        y = numpy.empty(x.shape, dtype=numpy.int32)
        y[x < 0] = 0
        y[(x >= 0) & (x <= 0.8)] = 1
        y[(x >= 0.8) & (x <= 1.5)] = 0
        y[x > 1.5] = 1
    else:
        raise ValueError("kind must be in (2, 3, 4).")
    x2 = numpy.random.rand(n)
    return numpy.vstack([x, x2]).T, y


def plot_ds(X, y, ax=None, title=None):
    """
    Plots a dataset, *X* is a dataset with two
    features, *y* contains the binary labels.
    """
    if ax is None:
        import matplotlib.pyplot as plt  # pragma: no cover

        ax = plt.gca()  # pragma: no cover
    colors = {0: "#88CCCC", 1: "#CCCC88"}
    c = [colors[_] for _ in y]
    ax.scatter(X[:, 0], X[:, 1], c=c, s=20, edgecolor="k", lw=0.1)
    if title is not None:
        ax.set_title(title)
    return ax


def plog2(p):
    """
    Computes :math:`x \\ln_2 x`.
    """
    if p == 0:
        return 0
    return p * numpy.log(p) / numpy.log(2)


def logistic(x):
    """
    Computes :math:`\\frac{1}{1 + e^{-x}}`.
    """
    return 1.0 / (1.0 + numpy.exp(-x))


def likelihood(x, y, theta=1.0, th=0.0):
    """
    Computes :math:`\\sum_i y_i f(\\theta (x_i - x_0)) +
    (1 - y_i) (1 - f(\\theta (x_i - x_0)))`
    where :math:`f(x_i)` is :math:`\\frac{1}{1 + e^{-x}}`.
    """
    lr = logistic((x - th) * theta)
    return y * lr + (1.0 - y) * (1 - lr)


def criteria(X, y):
    """
    Computes Gini, information gain, likelihood on a dataset
    with two features assuming the first coordinates is used to classify.

    @param      X       2D matrix
    @param      y       binary labels
    @return             dataframe
    """
    res = numpy.empty((X.shape[0], 8))
    res[:, 0] = X[:, 0]
    res[:, 1] = y
    order = numpy.argsort(res[:, 0])
    res = res[order, :].copy()
    x = res[:, 0].copy()
    y = res[:, 1].copy()

    for i in range(1, res.shape[0] - 1):
        # gini
        p1 = numpy.sum(y[:i]) / i
        p2 = numpy.sum(y[i:]) / (y.shape[0] - i)
        res[i, 2] = p1
        res[i, 3] = p2
        res[i, 4] = 1 - p1**2 - (1 - p1) ** 2 + 1 - p2**2 - (1 - p2) ** 2
        res[i, 5] = -plog2(p1) - plog2(1 - p1) - plog2(p2) - plog2(1 - p2)
        th = x[i]
        res[i, 6] = logistic(th)
        res[i, 7] = numpy.sum(likelihood(x, y, 1.0, th)) / res.shape[0]
    columns = ["X", "y", "p1", "p2", "Gini", "Gain", "lr", "LL"]
    return DataFrame(res[1:-1], columns=columns)


def criteria2(X, y):
    """
    Computes Gini, information gain, likelihood on a dataset
    with two features assuming the first coordinates is used to classify.

    @param      X       2D matrix
    @param      y       binary labels
    @return             dataframe
    """
    res = numpy.empty((X.shape[0], 5))
    res[:, 0] = X[:, 0]
    res[:, 1] = y
    order = numpy.argsort(res[:, 0])
    res = res[order, :].copy()
    x = res[:, 0].copy()
    y = res[:, 1].copy()

    for i in range(1, res.shape[0] - 1):
        # gini
        th = x[i]
        res[i, 2] = (
            max(
                numpy.sum(likelihood(x, y, 1.0, th)),
                numpy.sum(likelihood(x, y, -1.0, th)),
            )
            / res.shape[0]
        )
        res[i, 3] = (
            max(
                numpy.sum(likelihood(x, y, 10.0, th)),
                numpy.sum(likelihood(x, y, -10.0, th)),
            )
            / res.shape[0]
        )
        res[i, 4] = (
            max(
                numpy.sum(likelihood(x, y, 100.0, th)),
                numpy.sum(likelihood(x, y, -100.0, th)),
            )
            / res.shape[0]
        )
    columns = ["X", "y", "LL", "LL-10", "LL-100"]
    return DataFrame(res[1:-1], columns=columns)
