# -*- coding: utf-8 -*-
"""
@file
@brief Implements classic k-nn.
"""
import numpy
import numpy.linalg
from scipy.spatial.distance import euclidean


class NuagePoints:
    """
    Définit une classe de nuage de points.
    On suppose qu'ils sont définis par une matrice,
    chaque ligne est un élément.
    """

    def __init__(self):
        """
        constructeur
        """
        pass

    def fit(self, X, y=None):
        """
        Follows sklearn API.

        @param      X   training set
        @param      y   labels
        """
        self.nuage = X
        self.labels = y

    def kneighbors(self, X, n_neighbors=1, return_distance=True):
        """
        Return the k nearest neighbors.

        @param      X                   test set
        @param      n_neighbors         number of neighbors
        @param      return_distance     return distance as well
        @return                         array (dist), array (indices)
        """
        if n_neighbors != 1:
            raise NotImplementedError(  # pragma: no cover
                "Not implemented when n_neighbors != 1.")
        if not return_distance:
            raise NotImplementedError(  # pragma: no cover
                "Not implemented when return_distance is False.")

        dist = numpy.zeros(X.shape[0])
        ind = numpy.zeros(X.shape[0], dtype=numpy.int64)

        for i in range(X.shape[0]):
            row = X[i, :]
            r = self.ppv(row)
            dist[i], ind[i] = r
        return dist, ind

    @property
    def shape(self):
        """
        Retourne la dimension du nuage.
        """
        return self.nuage.shape

    def distance(self, obj1, obj2):
        """
        Retourne une distance entre deux éléments.

        @param      obj1        object 1
        @param      obj2        object 2
        @return                 distance
        """
        try:
            return euclidean(obj1, obj2)
        except ValueError as e:
            raise ValueError(
                f"Unable to compute euclidean distance with shapes "
                f"{obj1.shape} and {obj2.shape}.") from e

    def label(self, i):
        """
        Retourne le label de l'object d'indice ``i``.

        @param      i       indice
        @return             label or None if there is no label
        """
        return self.label[i] if self.label is not None else None

    def ppv(self, obj):
        """
        Retourne l'élément le plus proche de obj et sa distance avec obj.

        @param      obj     object
        @return             ``tuple(dist, index)``
        """
        if len(obj.shape) == 1:
            obj = obj.reshape((1, -1))
        ones = numpy.ones((self.nuage.shape[0], 1))
        mat = ones @ obj
        if len(mat.shape) == 1:
            mat.resize((mat.shape[0], 1))
        delta = self.nuage - mat
        norm = numpy.linalg.norm(delta, axis=1)
        i = numpy.argmin(norm)
        return norm[i], i
