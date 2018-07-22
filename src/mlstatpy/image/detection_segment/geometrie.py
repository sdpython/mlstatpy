# -*- coding: utf-8 -*-
"""
@file
@brief Définition de petits éléments géométriques tels que les points
et les segments, implemente également des opérations standard
telles le produit scalaire entre deux vecteurs, ...
"""
import math
import copy
import numpy


class Point:
    """
    Définit un point de l'image ou un vecteur,
    deux coordonnées *x* et *y* qui sont réelles.
    """
    __slots__ = "x", "y"

    def __init__(self, x, y):
        """constructeur"""
        self.x = x
        self.y = y

    def __str__(self):
        """permet d'afficher un point avec l'instruction print"""
        return '({0},{1})'.format(self.x, self.y)

    def __repr__(self):
        """usuel"""
        return Point(self.x, self.y)

    def normalise(self):
        """normalise le vecteur, sa norme devient 1"""
        v = self.x * self.x + self.y * self.y
        v = math.sqrt(v)
        if v > 0:   # evite les erreurs si sa norme est nulle
            self.x /= v
            self.y /= v

    def scalairek(self, k: float):
        """
        Mulitplication par un scalaire.

        @param      k       float
        """
        self.x *= k
        self.y *= k

    def norme(self) -> float:
        """
        Retourne la norme.

        @return         float (norm)
        """
        return math.sqrt(self.x * self.x + self.y * self.y)

    def as_array(self):
        """
        Convertit en array.
        """
        return numpy.array([self.x, self.y])

    def scalaire(self, k: 'Point') -> float:
        """
        Calcule le produit scalaire.

        @param      k       @see cl Point
        @return             float
        """
        return self.x * k.x + self.y * k.y

    def __iadd__(self, ad):
        """ajoute un vecteur à celui-ci"""
        self.x += ad.x
        self.y += ad.y
        return self

    def __add__(self, ad):
        """ajoute un vecteur a celui-ci"""
        return Point(self.x + ad.x, self.y + ad.y)

    def arrondi(self) -> 'Point':
        """
        retourne les coordonnées arrondies à l'entier le plus proche
        """
        return Point(int(self.x + 0.5), int(self.y + 0.5))

    def __sub__(self, p):
        """soustraction de deux de vecteurs"""
        return Point(self.x - p.x, self.y - p.y)

    def angle(self):
        """retourne l'angle du vecteur"""
        return math.atan2(self.y, self.x)

    def __eq__(self, a) -> bool:
        """retourne True si les deux points ``self`` et ``a`` sont egaux,
        False sinon"""
        return self.x == a.x and self.y == a.y


class Segment:
    """
    Définit un segment, soit deux @see cl Point.
    """

    # voir le commentaire associees a la ligne contenant __slots__
    # dans la classe Point
    __slots__ = "a", "b"

    def __init__(self, a, b):
        """
        constructeur, pour éviter des erreurs d'etourderie,
        on crée des copies des extrémités a et b,
        comme ce sont des classes, une simple affectation ne suffit pas
        """
        self.a, self.b = copy.copy(a), copy.copy(b)

    def __str__(self) -> str:
        """permet d'afficher le segment avec l'instruction print"""
        return "[{0},{1}]".format(self.a, self.b)

    def directeur(self) -> Point:
        """retourne le vecteur directeur du segment,
        ce vecteur est norme"""
        p = Point(self.b.x - self.a.x, self.b.y - self.a.y)
        p.normalise()
        return p

    def normal(self) -> float:
        """retourne le vecteur normal du segment,
        ce vecteur est norme"""
        p = Point(self.a.y - self.b.y, self.b.x - self.a.x)
        p.normalise()
        return p

    def first(self):
        """Retourne la première extrémité."""
        return self.a

    def last(self):
        """Retourne la seconde extrémité."""
        return self.b
