# -*- coding: utf-8 -*-

import random
import numpy
from .kppv import NuagePoints


class NuagePointsLaesa(NuagePoints):
    """
    Implémente l'algorithme des plus proches voisins,
    version :ref:`LAESA <space_metric_algo_laesa_prime>`.
    """

    def __init__(self, nb_pivots):
        """
        Construit la classe

        @param      nb_pivots       number of pivots
        """
        NuagePoints.__init__(self)
        self.nb_pivots = nb_pivots

    def fit(self, X, y=None):
        """
        Follows sklearn API.

        @param      X   training set
        @param      y   labels
        """
        self.nuage = X
        self.labels = y
        self.selection_pivots(self.nb_pivots)

    def selection_pivots(self, nb):
        """
        Sélectionne *nb* pivots aléatoirements.

        @param      nb      nombre de pivots
        """
        nb = min(nb, self.nuage.shape[0])
        if nb == 1:
            self.pivots = [2]
        else:
            self.pivots = set()
            while len(self.pivots) < nb:
                i = random.randint(0, self.nuage.shape[0] - 1)
                if i not in self.pivots:
                    self.pivots.add(i)
            self.pivots = list(sorted(self.pivots))

        # on calcule aussi la distance de chaque éléments au pivots
        self.dist = numpy.zeros((self.nuage.shape[0], len(self.pivots)))
        for i in range(self.nuage.shape[0]):
            for j in range(len(self.pivots)):  # pylint: disable=C0200
                self.dist[i, j] = self.distance(
                    self.nuage[i, :], self.nuage[self.pivots[j], :]
                )

    def ppv(self, obj):
        """
        Retourne l'élément le plus proche de obj et sa distance avec obj,
        utilise la sélection à l'aide pivots

        @param      obj     object
        @return             ``tuple(distance, index)``
        """

        # initialisation
        dp = [
            (self.distance(obj, self.nuage[p, :]), p, i)
            for i, p in enumerate(self.pivots)
        ]

        # pivots le plus proche
        dm, im, _ = min(dp)

        # améliorations
        for i in range(0, self.nuage.shape[0]):
            # on regarde si un pivot permet d'éliminer l'élément i
            calcul = True
            for d, p, ip in dp:
                delta = abs(d - self.dist[i, ip])
                if delta > dm:
                    calcul = False
                    break

            # dans le cas contraire on calcule la distance
            if calcul:
                d = self.distance(obj, self.nuage[i, :])
                if d < dm:
                    dm = d
                    im = i

        return dm, im
