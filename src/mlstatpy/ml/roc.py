# -*- coding: utf-8 -*-
"""
@file
@brief About ROC
"""
import copy
import random
import pandas
import math


class ROC:
    """
    Helper to draw a ROC curve

    .. todoext::
        :title: refactor ROC
        :tag: enhancement

        This code is very old (before pandas).
        It should be updated to be more efficient.
    """

    def __init__(self, df):
        """
        initialisation with a dataframe and two columns:

        * column 1: score
        * column 2: expected answer (boolean)
        * column 3: weight (optional)
        """
        if isinstance(df, list):
            if len(df[0]) == 2:
                self.data = pandas.DataFrame(df, columns=["score", "label"])
            else:
                self.data = pandas.DataFrame(df, columns=["score", "label", "weight"])
        elif not isinstance(df, pandas.DataFrame):
            raise TypeError(
                "df should be a DataFrame, not {0}".format(type(df)))
        else:
            self.data = df.copy()
        self.data.sort_values(self.data.columns[0], inplace=True)
        if self.data.shape[1] == 2:
            self.data["weight"] = 1.0

    def __len__(self):
        """
        usual
        """
        return len(self.data)

    def __str__(self):
        """
        show first elements
        """
        s = "TestROC reco rate : " + \
            (("%3.2f") % (self.reco_rate() * 100)) + "%\n"
        for i in range(0, min(5, len(self))):
            s += "      " + \
                str(i) + "\t" + str(self.data.ix[i][0]) + \
                "\t" + str(self.data.ix[i][1]) + "\n"
        for i in range(max(len(self) - 5, 0), len(self)):
            s += "      " + \
                str(i) + "\t" + str(self.data.ix[i][0]) + \
                "\t" + str(self.data.ix[i][1]) + "\n"
        s += "      ----------------------------------------------\n"
        roc = self.ROC(10, True)
        s += "      read rate\terror rate\n"
        for r in roc:
            s += "      " + ("%3.2f" % (r[0] * 100)) + \
                " %\t" + ("%3.2f" % (r[1] * 100)) + " %\n"
        s += "      ----------------------------------------------\n"
        roc = self.ROC(10, False)
        s += "      reco rate\t error rate\n"
        for r in roc:
            s += "      " + ("%3.2f" % (r[0] * 100)) + \
                " %\t" + ("%3.2f" % (r[1] * 100)) + " %\n"

        return s

    def reco_rate(self):
        """calcule le taux de reconnaissance"""
        nb = self.data[self.data.columns[1]].sum()
        return float(nb) / len(self)

    def ROC(self, nb=100, read=True, bootstrap=False):
        """
        calcule une courbe ROC avec nb points seuils, si nb == -1, autant de points de seuil que d'observations,
        si bootstrap == True, tire aléatoire des nombres pour créer une zone d'intervalle de confiance
        """
        if not bootstrap:
            cloud = self.data
        else:
            cloud = self.random_cloud()

        # sélection des seuils
        nb = min(nb, len(cloud))
        seuil = []
        for i in range(0, nb):
            j = len(cloud) * i // nb
            seuil.append(cloud.iloc[j, 0])

        # on trace la courbe
        roc = []
        s = len(seuil) - 1
        current = [0, 0]
        for ind in range(len(cloud) - 1, -1, -1):
            l = (cloud.iloc[ind, 0], cloud.iloc[ind, 1])
            if (l[0] < seuil[s]) and s > 0:
                roc.append(copy.copy(current))
                s -= 1
            current[0] += 1
            if not l[1]:
                current[1] += 1
        if current[0] != 0:
            roc.append(copy.copy(current))
        roc.reverse()

        # stat
        if read:
            for l in roc:
                if l[0] > 0:
                    l[1] = float(l[1]) / float(l[0])
                    l[0] = float(l[0]) / float(len(cloud))
        else:
            good, wrong = 0, 0
            for l in cloud:
                if l[1]:
                    good += 1
                else:
                    wrong += 1

            for l in roc:
                l[0] -= l[1]
                if good > 0:
                    l[0] = float(l[0]) / good
                if wrong > 0:
                    l[1] = float(l[1]) / wrong
        return roc

    def plot(self, nblist=[100], read=False, bootstrap=0, ax=None):
        """
        trace plusieurs courbes ROC sur le même dessin

        @param      nblist      number of points
        @param      read        if True, plot the reading rate, False, precision / recall
        @param      boostrap    number of curve for the boostrap (0 for None)
        @param      ax          axis
        @return                 ax
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        ncolor = ["red", "blue", "green", "black", "orange"]

        ax.set_xlabel("error rate")
        if read:
            ax.set_ylabel("read rate")
        else:
            ax.set_ylabel("reco rate")
        n = 0

        if bootstrap <= 0:
            for s in nblist:
                roc = self.ROC(s, read)
                x = [r[1] for r in roc]
                y = [r[0] for r in roc]
                c = ncolor[n % len(ncolor)]
                ax.plot(x, y, linewidth=1.0, color=c)
                n = n + 1
        else:
            for s in nblist:
                c = ncolor[n % len(ncolor)]
                for l in range(0, bootstrap):
                    roc = self.ROC(s, read, bootstrap=True)
                    x = [r[1] for r in roc]
                    y = [r[0] for r in roc]
                    ax.plot(x, y, linewidth=0.15, color=c)
                    n = n + 1
        return ax

    def ROC_point(self, roc, error):
        """
        détermine un point de la courbe passant par (error, y), y étant à détermine,
        la réponse est retourné par interpolation linéaire
        """
        for i in range(0, len(roc)):
            if roc[i][1] <= error:
                break

        if i == len(roc):
            return 0

        p2 = roc[i]
        if i - 1 > 0:
            p1 = [1, 1]
        else:
            p1 = roc[i - 1]

        rate = (error - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0]) + p1[0]
        return rate

    def ROC_point_intervalle(self, error, nb, read=True, bootstrap=10, alpha=0.05):
        """
        détermine un intervalle de confiance pour un taux de lecture pour un taux d'erreur donné,
        retourne un taux d'erreur et un intervalle de confiance, retourne aussi le couple min,max,
        cette troisème liste contient aussi moyenne, écart-type, médiance
        """

        rate = []
        for i in range(0, bootstrap):
            roc = self.ROC(nb, read, bootstrap=True)
            r = self.ROC_point(roc, error)
            rate.append(r)

        rate.sort()

        roc = self.ROC(nb, read)
        ra = self.ROC_point(roc, error)

        i1 = int(alpha * len(rate) / 2)
        i2 = int(min(1.0 - alpha / 2 * len(rate) + 0.5, len(rate) - 1))
        med = rate[len(rate) // 2]
        moy = float(sum(rate)) / len(rate)
        var = 0
        for r in rate:
            var += r * r
        var = float(var) / len(rate)
        var = var - moy * moy
        return ra, [rate[i1], rate[i2]], [rate[0], rate[len(rate) - 1], moy, math.sqrt(var), med]

    def random_cloud(self):
        """
        tire un nuage aléatoirement
        """
        cloud = []
        for i in range(0, len(self)):
            k = random.randint(0, len(self) - 1)
            cloud.append(self.data.ix[k, :])
        cloud = pandas.DataFrame(cloud, columns=["score", "label"])
        return cloud.sort_values(list(cloud.columns))

    def split_good_wrong(self, cloud):
        """
        retourne deux listes, bon et mauvais scores
        """
        good = []
        wrong = []
        for c in cloud:
            if c[1]:
                good.append(c[0])
            else:
                wrong.append(c[0])
        return good, wrong

    def compute_AUC(self, cloud):
        """
        calcule l'aire en-dessous de la courbe
        """
        good, wrong = self.split_good_wrong(cloud)
        good.sort()
        wrong.sort()
        auc = 0.0
        for b in wrong:
            for a in good:
                if a > b:
                    auc += 1.0
                elif a >= b:
                    auc += 0.5
        n = len(wrong) * len(good)
        if n > 0:
            auc /= float(n)
        return auc

    def ROC_AUC(self, error, nb, bootstrap=10, alpha=0.95):
        """
        détermine un intervalle de confiance pour l'aire en dessous de la courbe ROC' par la méthode bootstrap
        retourne un taux d'erreur et un intervalle de confiance, retourne aussi le couple min,max
        """

        rate = []
        for i in range(0, bootstrap):

            if bootstrap <= 0:
                cloud = self._log
            else:
                cloud = self.random_cloud()
            auc = self.compute_AUC(cloud)
            rate.append(auc)

        rate.sort()

        ra = self.compute_AUC(self._log)

        i1 = int(alpha * len(rate) / 2)
        i2 = int(min(1.0 - alpha / 2 * len(rate) + 0.5, len(rate) - 1))
        med = rate[len(rate) // 2]
        moy = float(sum(rate)) / len(rate)
        var = 0
        for r in rate:
            var += r * r
        var = float(var) / len(rate)
        var = var - moy * moy
        return ra, [rate[i1], rate[i2]], [rate[0], rate[len(rate) - 1], moy, math.sqrt(var), med]
