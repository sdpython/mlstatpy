# -*- coding: utf-8 -*-
"""
@file
@brief About ROC
"""
import copy
import random
import pandas
import numpy
import math
import itertools


class ROC:
    """
    Helper to draw a ROC curve
    """

    def __init__(self, df):
        """
        initialisation with a dataframe and two columns:

        * column 1: score
        * column 2: expected answer (boolean)
        * column 3: weight (optional)

        @param  df      dataframe or array or list, it must contains 2 or 3 columns always in the same order
        """
        if isinstance(df, list):
            if len(df[0]) == 2:
                self.data = pandas.DataFrame(df, columns=["score", "label"])
            else:
                self.data = pandas.DataFrame(
                    df, columns=["score", "label", "weight"])
        elif isinstance(df, numpy.ndarray):
            if df.shape[1] == 2:
                self.data = pandas.DataFrame(df, columns=["score", "label"])
            else:
                self.data = pandas.DataFrame(
                    df, columns=["score", "label", "weight"])
        elif not isinstance(df, pandas.DataFrame):
            raise TypeError(
                "df should be a DataFrame, not {0}".format(type(df)))
        else:
            self.data = df.copy()
        self.data.sort_values(self.data.columns[0], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        if self.data.shape[1] == 2:
            self.data["weight"] = 1.0

    @property
    def Data(self):
        """
        returns the underlying dataframe
        """
        return self.data

    def __len__(self):
        """
        usual
        """
        return len(self.data)

    def __repr__(self):
        """
        show first elements, precision rate
        """
        return self.__str__()

    def __str__(self):
        """
        show first elements, precision rate
        """
        rows = []
        rows.append("Overall precision: %3.2f - AUC=%f" %
                    (self.precision(), self.auc()))
        rows.append("--------------")
        rows.append(str(self.data.head(min(5, len(self)))))
        rows.append("--------------")
        rows.append(str(self.data.tail(min(5, len(self)))))
        rows.append("--------------")
        roc = self.compute_roc_curve(10, False)
        rows.append(str(roc))
        rows.append("--------------")
        roc = self.compute_roc_curve(10, True)
        rows.append(str(roc))
        return "\n".join(rows)

    def precision(self):
        """
        Compute precision
        """
        score, weight = self.data.columns[0], self.data.columns[2]
        return (self.data[score] * self.data[weight] * 1.0).sum() / self.data[weight].sum()

    def compute_roc_curve(self, nb=100, recprec=True, bootstrap=False):
        """
        Compute a ROC curve with *nb* points avec nb,
        if *nb == -1*, there are as many as points as the data contains,
        if *bootstrap == True*, it draws random number to create confidence interval based on bootstrap method.

        @param      nb          number of points for the curve
        @param      recprec     precision/recall or true ROC curve
        @param      boostrap    builds the curve after resampling
        """
        if not bootstrap:
            cloud = self.data.copy()
        else:
            cloud = self.random_cloud()

        sum_weights = cloud[cloud.columns[2]].sum()
        nb = min(nb, len(cloud))
        seuil = numpy.arange(nb + 1) * sum_weights / nb

        cloud = cloud.iloc[::-1].copy()
        cloud["lw"] = cloud[cloud.columns[2]] * cloud[cloud.columns[1]]
        cloud["cw"] = cloud[cloud.columns[2]].cumsum()
        cloud["clw"] = cloud["lw"].cumsum()

        roc = pandas.DataFrame(0, index=numpy.arange(
            nb + 1), columns=["recall", "precision"])
        pos_roc = 0
        pos_seuil = 0

        if recprec:
            for i in range(len(cloud)):
                if cloud.iloc[i, 4] > seuil[pos_seuil]:
                    roc.iloc[pos_roc, 0] = cloud.iloc[i, 4] / sum_weights
                    roc.iloc[pos_roc, 1] = cloud.iloc[i, 5] / cloud.iloc[i, 4]
                    pos_roc += 1
                    pos_seuil += 1
            while pos_roc < len(roc):
                roc.iloc[pos_roc, 0] = cloud.iloc[-1, 4] / sum_weights
                roc.iloc[pos_roc, 0] = cloud.iloc[-1, 5] / cloud.iloc[-1, 4]
                pos_roc += 1
        else:
            roc.columns = ["Error Rate", "Recognition Rate"]
            sum_good_weights = cloud.iloc[-1, 5]
            sum_bad_weights = sum_weights - sum_good_weights
            for i in range(len(cloud)):
                if cloud.iloc[i, 4] > seuil[pos_seuil]:
                    roc.iloc[pos_roc, 0] = (
                        cloud.iloc[i, 4] - cloud.iloc[i, 5]) / sum_bad_weights
                    roc.iloc[pos_roc, 1] = cloud.iloc[i, 5] / sum_good_weights
                    pos_roc += 1
                    pos_seuil += 1
            while pos_roc < len(roc):
                roc.iloc[pos_roc, 0] = (
                    cloud.iloc[-1, 4] - cloud.iloc[-1, 5]) / sum_bad_weights
                roc.iloc[pos_roc, 1] = cloud.iloc[-1, 5] / sum_good_weights
                pos_roc += 1
        return roc

    def random_cloud(self):
        """
        resample among the data

        @return      DataFrame
        """
        res = self.data.sample(len(self.data), weights=self.data[
                               self.data.columns[2]], replace=True)
        return res.sort_values(res.columns[0])

    def plot(self, nb=100, recprec=False, bootstrap=0, ax=None, label=None, **kwargs):
        """
        plot a ROC curve

        @param      nb          number of points
        @param      read        if True, plot the reading rate, False, precision / recall
        @param      boostrap    number of curves for the boostrap (0 for None)
        @param      label       label of the curve
        @param      ax          axis
        @param      kwargs      sent to `pandas.plot <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html>`_
        @return                 ax

        If *label* is None, it only plots bootstrapped curves to represent the confidence
        inteval. If *label* is not None, the main curve is plotted in all cases.
        """
        if bootstrap > 0:
            ckwargs = kwargs.copy()
            if 'color' not in ckwargs:
                ckwargs['color'] = 'r'
            if 'linewidth' not in kwargs:
                ckwargs['linewidth'] = 0.2
            ckwargs['legend'] = False
            if 'label' in ckwargs:
                del ckwargs['label']
            for l in range(0, bootstrap):
                roc = self.compute_roc_curve(
                    nb, recprec=recprec, bootstrap=True)
                ax = roc.plot(x=roc.columns[0], y=roc.columns[
                              1], ax=ax, **ckwargs)

        if bootstrap <= 0 or label is not None:
            if 'legend' not in kwargs:
                kwargs['legend'] = False
            roc = self.compute_roc_curve(nb, recprec=recprec)
            if label is not None:
                memo = roc.columns[1]
                roc.columns = [roc.columns[0], label]
            ax = roc.plot(x=roc.columns[0], y=roc.columns[1], ax=ax, **kwargs)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[-1:], labels[-1:])
            if label is not None:
                ax.set_ylabel(memo)

        return ax

    def auc(self, cloud=None):
        """
        computes the area under the curve

        @param      cloud       data or None to use ``self.data``, the function
                                assumes the data is sorted
        @return                 AUC
        """
        if cloud is None:
            cloud = self.data
        good = cloud[cloud[cloud.columns[1]] == 1]
        wrong = cloud[cloud[cloud.columns[1]] == 0]
        auc = 0.0
        for a, b in itertools.product(good.itertuples(False), wrong.itertuples(False)):
            if a[0] > b[0]:
                auc += a[2] * b[2]
            elif a[0] >= b[0]:
                auc += a[2] * b[2] / 2
        if auc == 0 and good.shape[0] + wrong.shape[0] < self.data.shape[0]:
            raise ValueError("Label are not right, expect 0 and 1 not {0}".format(
                set(cloud[cloud.columns[1]])))
        n = len(wrong) * len(good)
        if n > 0:
            auc /= float(n)
        return auc

    def auc_interval(self, bootstrap=10, alpha=0.95):
        """
        Determines a confidence interval for the AUC with bootstrap.

        @param      bootstrap       number of random estimation
        @param      alpha           define the confidence interval
        @return                     dictionary of values
        """
        if bootstrap <= 1:
            raise ValueError("Use auc instead, bootstrap < 2")
        rate = []
        for i in range(0, bootstrap):
            cloud = self.random_cloud()
            auc = self.auc(cloud)
            rate.append(auc)

        rate.sort()
        ra = self.auc(self.data)

        i1 = int(alpha * len(rate) / 2)
        i2 = max(i1, len(rate) - i1 - 1)
        med = rate[len(rate) // 2]
        moy = float(sum(rate)) / len(rate)
        var = 0
        for r in rate:
            var += r * r
        var = float(var) / len(rate)
        var = var - moy * moy
        return dict(auc=ra, interval=(rate[i1], rate[i2]),
                    min=rate[0], max=rate[len(rate) - 1],
                    mean=moy, var=math.sqrt(var), mediane=med)

    def roc_intersect(self, roc, x):
        """
        ROC curve is defined by a set of points.
        This function interpolates those points to determine
        *y* for any *x*.

        @param      roc     ROC curve
        @param      x       x
        @return             y
        """
        below = roc[roc[roc.columns[0]] <= x]
        i = len(below)
        if i == len(roc):
            return 0.0

        p2 = tuple(roc.iloc[i, :])
        if i - 1 <= 0:
            p1 = (1, 1)
        else:
            p1 = tuple(roc.iloc[i - 1, :])

        if p1[0] == p2[0]:
            return (p1[1] + p2[0]) / 2
        else:
            return (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]

    def roc_intersect_interval(self, x, nb, recprec=True, bootstrap=10, alpha=0.05):
        """
        computes a confidence interval for the value returned by
        @see me roc_intersect.

        @param      roc     ROC curve
        @param      x       x
        @return             dictionary
        """

        rate = []
        for i in range(0, bootstrap):
            roc = self.compute_roc_curve(nb, recprec, bootstrap=True)
            r = self.roc_intersect(roc, x)
            rate.append(r)

        rate.sort()

        roc = self.compute_roc_curve(nb, recprec)
        ra = self.roc_intersect(roc, x)

        i1 = int(alpha * len(rate) / 2)
        i2 = max(i1, len(rate) - i1 - 1)
        med = rate[len(rate) // 2]
        moy = float(sum(rate)) / len(rate)
        var = 0
        for r in rate:
            var += r * r
        var = float(var) / len(rate)
        var = var - moy * moy
        return dict(y=ra, interval=(rate[i1], rate[i2]),
                    min=rate[0], max=rate[len(rate) - 1],
                    mean=moy, var=math.sqrt(var), mediane=med)
