# -*- coding: utf-8 -*-
"""
@file
@brief Ce module définit un segment qui va parcourir l'image,
en plus d'être un segment, cette classe inclut la dimension de l'image,
et une fonction repérant sur ce segment les gradients presque
orthogonaux à l'image.
"""

import numpy
import math
import copy
# import detection_nfa as DN
from .geometrie import Segment, Point
# import detection_param as DP

# param = DP.Parametre ()


class SegmentBord_Commun (Segment):
    """
    Définit un segment allant d'un bord a un autre de l'image,
    la methode importante est decoupe_gradient

        dim     est la dimension de l'image"""

    # voir la remarque dans la classe Point a propos de __slots__
    __slots__ = "dim"

    def __init__(self, dim):
        """constructeur, definit la definition de l'image"""
        Segment.__init__(self, Point(0, 0), Point(0, 0))
        self.dim = dim
        self.premier()

    def copy(self):
        """
        Copie l'instance.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """permet d'afficher le segment"""
        s = Segment.__str__(self)
        s += " -- dim -- " + self.dim.__str__()
        return s

    def decoupe_gradient(self, gradient, cos_angle, ligne_gradient, seuil_norme):
        """
        Pour un segment donne joignant deux bords de l'image,
        cette fonction recupere le gradient et construit une liste
        contenant des informations pour un pixel sur deux du segment,

        * norme* : memorise la norme du gradient en ce point de l'image
        * *pos* : memorise la position du pixel
        * *aligne* : est vrai si le gradient est presque orthogonal au segment,
          ce resultat est relie au parametre proba_bin,
          deux vecteurs sont proches en terme de direction,
          s'ils font partie du secteur angulaire defini par proba_bin

        Le parcours du segment commence à son origine ``self.a``,
        et on ajoute à chaque iteration deux fois le vecteur normal
        jusqu'a sortir du cadre de l'image,
        les informations sont stockées dans ``ligne_gradient`` qui a une liste
        d'informations préalablement créée au debut du programme
        de facon à gagner du temps.
        """
        n = self.directeur()
        nor = self.normal()
        n.scalairek(2.0)
        p = copy.copy(self.a)
        a = p.arrondi()

        i = 0
        while a.x >= 0 and a.y >= 0 and a.x < self.dim.x and a.y < self.dim.y:
            # on recupere l'element dans ligne ou doivent etre stockees les informations (ligne_gradient)
            t = ligne_gradient.info_ligne[i]

            # on recupere le gradient de l'image au pixel a
            g = gradient[(a.x, a.y)]

            # on calcul sa norme
            t.norme = g.norme()

            # on place les coordonnees du pixel dans t
            t.pos.x = p.x
            t.pos.y = p.y

            # si la norme est positive, le gradient a une direction
            # on regarde s'il est dans le meme secteur angulaire (proba_bin)
            # que le vecteur normal au segment (nor)
            if t.norme > seuil_norme:
                t.aligne = g.scalaire(nor) > cos_angle * t.norme
            else:
                t.aligne = False

            # on passe au pixel suivant
            p += n
            a = p.arrondi()   # calcul de l'arrondi
            i += 1

        # on indique a ligne_gradient le nombre de pixel pris en compte
        # ensuite, on decidera si ce segment est effectivement un segment de l'image
        ligne_gradient.nb = i
