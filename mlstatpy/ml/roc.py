# -*- coding: utf-8 -*-

import math
import itertools
from enum import Enum
import pandas
import numpy


class ROC:
    """
    Helper to draw a :epkg:`ROC` curve.
    """

    class CurveType(Enum):
        """
        Curve types:

        * *PROBSCORE*: 1 - False Positive / True Positive
        * *ERRPREC*: error / recall
        * *RECPREC*: precision / recall
        * *ROC*: False Positive / True Positive
        * *SKROC*: False Positive / True Positive
          (`scikit-learn
          <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html>`_)
        """

        PROBSCORE = 2
        ERRREC = 3
        RECPREC = 4
        ROC = 5
        SKROC = 6

    def __init__(self, y_true=None, y_score=None, sample_weight=None, df=None):
        """
        Initialisation with a dataframe and two or three columns:

        * column 1: score (y_score)
        * column 2: expected answer (boolean) (y_true)
        * column 3: weight (optional) (sample_weight)

        @param  y_true          if *df* is None, *y_true*, *y_score*,
                                *sample_weight* must be filled,
                                *y_true* is whether or None the answer is true.
                                *y_true* means the prediction is right.
        @param  y_score         score prediction
        @param  sample_weight   weights
        @param  df              dataframe or array or list,
                                it must contains 2 or 3 columns always in the same order
        """
        if df is None:
            df = pandas.DataFrame()
            df["score"] = y_score
            df["label"] = y_true
            if sample_weight is not None:
                df["weight"] = sample_weight
            self.data = df
        elif isinstance(df, list):
            if len(df[0]) == 2:
                self.data = pandas.DataFrame(df, columns=["score", "label"])
            else:
                self.data = pandas.DataFrame(df, columns=["score", "label", "weight"])
        elif isinstance(df, numpy.ndarray):
            if df.shape[1] == 2:
                self.data = pandas.DataFrame(df, columns=["score", "label"])
            else:
                self.data = pandas.DataFrame(df, columns=["score", "label", "weight"])
        elif not isinstance(df, pandas.DataFrame):
            raise TypeError(  # pragma: no cover
                f"df should be a DataFrame, not {type(df)}"
            )
        else:
            self.data = df.copy()
        self.data.sort_values(self.data.columns[0], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        if self.data.shape[1] == 2:
            self.data["weight"] = 1.0

    @property
    def Data(self):
        """
        Returns the underlying dataframe.
        """
        return self.data

    def __len__(self):
        """
        usual
        """
        return len(self.data)

    def __repr__(self):
        """
        Shows first elements, precision rate.
        """
        return self.__str__()

    def __str__(self):
        """
        Shows first elements, precision rate.
        """
        rows = []
        rows.append(f"Overall precision: {self.precision():3.2f} - AUC={self.auc():f}")
        rows.append("--------------")
        rows.append(str(self.data.head(min(5, len(self)))))
        rows.append("--------------")
        rows.append(str(self.data.tail(min(5, len(self)))))
        rows.append("--------------")
        roc = self.compute_roc_curve(10, ROC.CurveType.ROC)
        rows.append(str(roc))
        rows.append("--------------")
        roc = self.compute_roc_curve(10, ROC.CurveType.ERRREC)
        rows.append(str(roc))
        return "\n".join(rows)

    def confusion(self, score=None, nb=10, curve=CurveType.ROC, bootstrap=False):
        """
        Computes the confusion matrix for a specific *score*
        or all if *score* is None.

        @param      score       score or None.
        @param      nb          number of scores (if *score* is None)
        @param      curve       see :class:`CurveType <mlstatpy.ml.roc.ROC.CurveType>`
        @param      boostrap    builds the curve after resampling
        @return                 One row if score is precised, many roww is score is None
        """
        if not bootstrap:
            cloud = self.data.copy()
        else:
            cloud = self.random_cloud()

        if score is None:
            sum_weights = cloud[cloud.columns[2]].sum()
            if nb <= 0:
                nb = len(cloud)
            else:
                nb = min(nb, len(cloud))
            seuil = numpy.arange(nb + 1) * sum_weights / nb

            cloud = cloud.iloc[::-1].copy()
            cloud["lw"] = cloud[cloud.columns[2]] * cloud[cloud.columns[1]]
            cloud["cw"] = cloud[cloud.columns[2]].cumsum()
            cloud["clw"] = cloud["lw"].cumsum()
            if cloud.columns[4] != "cw":
                raise ValueError("Column 4 should be 'cw'.")  # pragma: no cover
            if cloud.columns[5] != "clw":
                raise ValueError("Column 5 should be 'clw'.")  # pragma: no cover

            pos_roc = 0
            pos_seuil = 0
            if curve is ROC.CurveType.ROC:
                roc = pandas.DataFrame(
                    0,
                    index=numpy.arange(nb + 2),
                    columns=[
                        "True Positive",
                        "False Positive",
                        "False Negative",
                        "True Negative",
                        "threshold",
                    ],
                )
                sum_good_weights = cloud.iloc[-1, 5]
                sum_bad_weights = sum_weights - sum_good_weights
                roc.iloc[0, 0] = 0
                roc.iloc[0, 1] = 0
                roc.iloc[0, 2] = sum_good_weights
                roc.iloc[0, 3] = sum_bad_weights
                roc.iloc[0, 4] = max(cloud.iloc[:, 0])
                pos_roc += 1
                for i in range(len(cloud)):
                    if cloud.iloc[i, 4] > seuil[pos_seuil]:
                        tp = cloud.iloc[i, 5]
                        fp = cloud.iloc[i, 4] - cloud.iloc[i, 5]
                        roc.iloc[pos_roc, 0] = tp
                        roc.iloc[pos_roc, 1] = fp
                        roc.iloc[pos_roc, 2] = sum_good_weights - tp
                        roc.iloc[pos_roc, 3] = sum_bad_weights - fp
                        roc.iloc[pos_roc, 4] = cloud.iloc[i, 0]
                        pos_roc += 1
                        pos_seuil += 1
                roc.iloc[pos_roc:, 0] = sum_good_weights
                roc.iloc[pos_roc:, 1] = sum_bad_weights
                roc.iloc[pos_roc:, 2] = 0
                roc.iloc[pos_roc:, 3] = 0
                roc.iloc[pos_roc:, 4] = min(cloud.iloc[:, 0])
                return roc
            raise NotImplementedError(  # pragma: no cover
                f"Unexpected type '{curve}', only ROC is allowed."
            )

        # if score is not None
        roc = self.confusion(nb=len(self), curve=curve, bootstrap=False, score=None)
        roc = roc[roc["threshold"] <= score]
        if len(roc) == 0:
            raise ValueError(  # pragma: no cover
                f"The requested confusion is empty for score={score}."
            )
        return roc[:1]

    def precision(self):
        """
        Computes the precision.
        """
        score, weight = self.data.columns[0], self.data.columns[2]
        return (self.data[score] * self.data[weight] * 1.0).sum() / self.data[
            weight
        ].sum()

    def compute_roc_curve(self, nb=100, curve=CurveType.ROC, bootstrap=False):
        """
        Computes a ROC curve with *nb* points avec nb,
        if *nb == -1*, there are as many as points as the data contains,
        if *bootstrap == True*, it draws random number to create confidence
        interval based on bootstrap method.

        @param      nb          number of points for the curve
        @param      curve       see :class:`CurveType <mlstatpy.ml.roc.ROC.CurveType>`
        @param      bootstrap   builds the curve after resampling
        @return                 DataFrame (metrics and threshold)

        If *curve* is *SKROC*, the parameter *nb* is not taken into account.
        It should be set to 0.
        """
        if curve is ROC.CurveType.ERRREC:
            roc = self.compute_roc_curve(
                nb=nb, curve=ROC.CurveType.RECPREC, bootstrap=bootstrap
            )
            roc["error"] = -roc["precision"] + 1
            return roc[["error", "recall", "threshold"]]
        if curve is ROC.CurveType.PROBSCORE:
            roc = self.compute_roc_curve(
                nb=nb, curve=ROC.CurveType.ROC, bootstrap=bootstrap
            )
            roc["P(->s)"] = roc["False Positive Rate"]
            roc["P(+<s)"] = -roc["True Positive Rate"] + 1
            return roc[["P(+<s)", "P(->s)", "threshold"]]

        if not bootstrap:
            cloud = self.data.copy()
        else:
            cloud = self.random_cloud()

        if curve is ROC.CurveType.SKROC:
            if nb > 0:
                raise NotImplementedError(  # pragma: no cover
                    "nb must be <= 0 si curve is SKROC"
                )
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(
                y_true=cloud[cloud.columns[1]],
                y_score=cloud[cloud.columns[0]],
                sample_weight=cloud[cloud.columns[2]],
            )
            roc = pandas.DataFrame(
                0,
                index=numpy.arange(len(fpr)),
                columns=["False Positive Rate", "True Positive Rate", "threshold"],
            )
            roc_cols = list(roc.columns)
            roc[roc_cols[0]] = fpr
            roc[roc_cols[1]] = tpr
            roc[roc_cols[2]] = thresholds
            return roc

        sum_weights = cloud[cloud.columns[2]].sum()
        if nb <= 0:
            nb = len(cloud)
        else:
            nb = min(nb, len(cloud))
        seuil = numpy.arange(nb + 1) * sum_weights / nb

        cloud = cloud.iloc[::-1].copy()
        cloud["lw"] = cloud[cloud.columns[2]] * cloud[cloud.columns[1]]
        cloud["cw"] = cloud[cloud.columns[2]].cumsum()
        cloud["clw"] = cloud["lw"].cumsum()
        sum_weights_ans = cloud["lw"].sum()
        if cloud.columns[4] != "cw":
            raise ValueError("Column 4 should be 'cw'.")  # pragma: no cover
        if cloud.columns[5] != "clw":
            raise ValueError("Column 5 should be 'clw'.")  # pragma: no cover

        pos_roc = 0
        pos_seuil = 0

        if curve is ROC.CurveType.ROC:
            roc = pandas.DataFrame(
                0,
                index=numpy.arange(nb + 1),
                columns=["False Positive Rate", "True Positive Rate", "threshold"],
            )
            sum_good_weights = cloud.iloc[-1, 5]
            sum_bad_weights = sum_weights - sum_good_weights
            for i in range(len(cloud)):
                if cloud.iloc[i, 4] > seuil[pos_seuil]:
                    ds = cloud.iloc[i, 4] - cloud.iloc[i, 5]
                    roc.iloc[pos_roc, 0] = ds / sum_bad_weights
                    roc.iloc[pos_roc, 1] = cloud.iloc[i, 5] / sum_good_weights
                    roc.iloc[pos_roc, 2] = cloud.iloc[i, 0]
                    pos_roc += 1
                    pos_seuil += 1
            roc.iloc[pos_roc:, 0] = (
                cloud.iloc[-1, 4] - cloud.iloc[-1, 5]
            ) / sum_bad_weights
            roc.iloc[pos_roc:, 1] = cloud.iloc[-1, 5] / sum_good_weights
            roc.iloc[pos_roc:, 2] = cloud.iloc[-1, 0]

        elif curve is ROC.CurveType.RECPREC:
            roc = pandas.DataFrame(
                0,
                index=numpy.arange(nb + 1),
                columns=["recall", "precision", "threshold"],
            )
            for i in range(len(cloud)):
                if cloud.iloc[i, 4] > seuil[pos_seuil]:
                    roc.iloc[pos_roc, 0] = cloud.iloc[i, 4] / sum_weights
                    if cloud.iloc[i, 4] > 0:
                        roc.iloc[pos_roc, 1] = cloud.iloc[i, 5] / cloud.iloc[i, 4]
                    else:
                        roc.iloc[pos_roc, 1] = 0.0
                    roc.iloc[pos_roc, 2] = cloud.iloc[i, 0]
                    pos_roc += 1
                    pos_seuil += 1
            roc.iloc[pos_roc:, 0] = 1.0
            roc.iloc[pos_roc:, 1] = sum_weights_ans / sum_weights
            roc.iloc[pos_roc:, 2] = cloud.iloc[-1, 0]

        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unknown curve type '{curve}'."
            )

        return roc

    def random_cloud(self):
        """
        Resamples among the data.

        @return      DataFrame
        """
        res = self.data.sample(
            len(self.data), weights=self.data[self.data.columns[2]], replace=True
        )
        return res.sort_values(res.columns[0])

    def plot(
        self,
        nb=100,
        curve=CurveType.ROC,
        bootstrap=0,
        ax=None,
        thresholds=False,
        **kwargs,
    ):
        """
        Plots a :epkg:`ROC` curve.

        @param      nb          number of points
        @param      curve       see :class:`CurveType <mlstatpy.ml.roc.ROC.CurveType>`
        @param      bootstrap   number of curves for the boostrap (0 for None)
        @param      ax          axis
        @param      thresholds  use thresholds for the X axis
        @param      kwargs      sent to `pandas.plot <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html>`_
        @return                 ax
        """
        nb_bootstrap = 0
        if bootstrap > 0:
            ckwargs = kwargs.copy()
            if "color" not in ckwargs:
                ckwargs["color"] = "r"
            if "linewidth" not in kwargs:
                ckwargs["linewidth"] = 0.2
            ckwargs["legend"] = False
            if "label" in ckwargs:
                del ckwargs["label"]
            for _ in range(0, bootstrap):
                roc = self.compute_roc_curve(nb, curve=curve, bootstrap=True)
                if thresholds:
                    cols = list(_ for _ in roc.columns if _ != "threshold")
                    roc = roc.sort_values("threshold").reset_index(drop=True)
                    ax = roc.plot(
                        x="threshold",
                        y=cols,
                        ax=ax,
                        label=["_nolegend_" for i in cols],
                        **ckwargs,
                    )
                else:
                    cols = list(_ for _ in roc.columns[1:] if _ != "threshold")
                    roc = roc.sort_values(roc.columns[0]).reset_index(drop=True)
                    ax = roc.plot(
                        x=roc.columns[0],
                        y=cols,
                        ax=ax,
                        label=["_nolegend_" for i in cols],
                        **ckwargs,
                    )
                nb_bootstrap += len(cols)
            bootstrap = 0

        if bootstrap <= 0:
            if "legend" not in kwargs:
                kwargs["legend"] = False
            roc = self.compute_roc_curve(nb, curve=curve)
            if not thresholds:
                roc = roc[[_ for _ in roc.columns if _ != "threshold"]]

            cols = list(_ for _ in roc.columns if _ != "threshold")
            final = 0
            if thresholds:
                if "label" in kwargs and len(cols) != len(kwargs["label"]):
                    raise ValueError(  # pragma: no cover
                        f"label must have {len(cols)} values"
                    )
                roc = roc.sort_values("threshold").reset_index(drop=True)
                ax = roc.plot(x="threshold", y=cols, ax=ax, **kwargs)
                ax.set_ylim([0, 1])
                ax.set_xlabel("thresholds")
                final += len(cols)
                diag = 0
            else:
                if "label" in kwargs and len(cols) - 1 != len(kwargs["label"]):
                    raise ValueError(f"label must have {len(cols) - 1} values")
                final += len(cols) - 1
                roc = roc.sort_values(cols[0]).reset_index(drop=True)
                ax = roc.plot(x=cols[0], y=cols[1:], ax=ax, **kwargs)
                if curve is ROC.CurveType.ROC or curve is ROC.CurveType.SKROC:
                    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                    diag = 1
                else:
                    diag = 0
                ax.set_xlabel(roc.columns[0])
                if len(roc.columns) == 2:
                    ax.set_ylabel(roc.columns[1])
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            # legend
            handles, labels = ax.get_legend_handles_labels()
            tot = final + nb_bootstrap
            diag = len(handles) - diag
            handles = handles[:-tot] + handles[-final:diag]
            new_labels = labels[:-tot] + labels[-final:diag]
            ax.legend(handles, new_labels)

        return ax

    def auc(self, cloud=None):
        """
        Computes the area under the curve (:epkg:`AUC`).

        @param      cloud       data or None to use ``self.data``, the function
                                assumes the data is sorted.
        @return                 AUC

        The first column is the label, the second one is the score,
        the third one is the weight.
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
            raise ValueError(  # pragma: no cover
                "Label are not right, expect 0 and 1 not {0}".format(
                    set(cloud[cloud.columns[1]])
                )
            )
        n = len(wrong) * len(good)
        if n > 0:
            auc /= float(n)
        return auc

    def auc_interval(self, bootstrap=10, alpha=0.95):
        """
        Determines a confidence interval for the :epkg:`AUC` with bootstrap.

        @param      bootstrap       number of random estimations
        @param      alpha           define the confidence interval
        @return                     dictionary of values
        """
        if bootstrap <= 1:
            raise ValueError("Use auc instead, bootstrap < 2")  # pragma: no cover
        rate = []
        for _ in range(0, bootstrap):
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
        return dict(
            auc=ra,
            interval=(rate[i1], rate[i2]),
            min=rate[0],
            max=rate[len(rate) - 1],
            mean=moy,
            var=math.sqrt(var),
            mediane=med,
        )

    def roc_intersect(self, roc, x):
        """
        The :epkg:`ROC` curve is defined by a set of points.
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
        return (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]

    def roc_intersect_interval(
        self, x, nb, curve=CurveType.ROC, bootstrap=10, alpha=0.05
    ):
        """
        Computes a confidence interval for the value returned by
        @see me roc_intersect.

        @param      x           x
        @param      nb          number of curves to draw
        @param      curve       see :class:`CurveType <mlstatpy.ml.roc.ROC.CurveType>`
        @param      bootstrap   number of random estimations
        @param      alpha       confidence interval
        @return                 dictionary
        """

        rate = []
        for _ in range(0, bootstrap):
            roc = self.compute_roc_curve(nb, curve=curve, bootstrap=True)
            r = self.roc_intersect(roc, x)
            rate.append(r)

        rate.sort()

        roc = self.compute_roc_curve(nb, curve=curve)
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
        return dict(
            y=ra,
            interval=(rate[i1], rate[i2]),
            min=rate[0],
            max=rate[len(rate) - 1],
            mean=moy,
            var=math.sqrt(var),
            mediane=med,
        )
