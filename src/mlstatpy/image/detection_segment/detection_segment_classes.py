# -*- coding: utf-8 -*-
"""
@file
@brief module inusite, inclut une second parcours des segments d'une image,
le segment va relier tous les couples de points du contour, sauf s'ils
appartiennent au meme bord"""
import numpy
import math
import copy
# import detection_nfa as DN
from .detection_segment_bord import SegmentBord_Commun


class SegmentBord (SegmentBord_Commun):
    """
    Définit un segment allant d'un bord à un autre de l'image,
    *dim* est la dimension de l'image.
    """
    __slots__ = "bord1", "bord2"

    def __init__(self, dim):
        """
        @param      dim     dimension de l'image
        """
        SegmentBord_Commun.__init__(self, dim)
        self.premier()

    def copy(self):
        """
        Copie l'instance.
        """
        return copy.deepcopy(self)

    def premier(self):
        """
        définit le premier segment
        """
        self.a.x, self.a.y = 0, 0
        self.b.x, self.b.y = self.dim.x - 1, 0
        self.bord1 = 0
        self.bord2 = 1

    def __str__(self):
        """
        usual
        """
        s = SegmentBord_Commun.__str__(self)
        s += " -- bord " + str(self.bord1) + "," + str(self.bord2)
        return s

    def next(self):
        """
        Passe au segment suivant, return False si on est dernier.
        """
        if self.bord2 == 1:
            if self.b.y < self.dim.y - 1:
                self.b.y += 1
            else:
                self.bord2 = 2
                self.b.x = self.dim.x - 2
        elif self.bord2 == 2:
            if self.b.x > 0:
                self.b.x -= 1
            else:
                self.b.y = self.dim.y - 2
                self.bord2 = 3
        elif self.bord2 == 3:
            if self.b.y > 0:
                self.b.y -= 1
            else:
                self.b.x = 1
                self.bord2 = 0
        elif self.bord2 == 0:
            if self.b.x < self.dim.x - 1:
                self.b.x += 1
            else:
                self.bord2 = 1
                self.b.y = 1

        if self.a.x == self.b.x and self.a.y == self.b.y:
            r = self.nexta()
            if not r:
                return r
            if self.a.x == self.b.x:
                if self.b.y != 0 and self.a.y != 0:
                    return self.next()
                elif self.b.y != self.dim.y - 1 and self.a.y != self.dim.y - 1:
                    return self.next()
            elif self.a.y == self.b.y:
                if self.b.x != 0 and self.a.x != 0:
                    return self.next()
                elif self.b.x != self.dim.x - 1 and self.a.x != self.dim.x - 1:
                    return self.next()
            return r
        else:
            if self.a.x == self.b.x:
                if self.b.y != 0 and self.a.y != 0:
                    return self.next()
                elif self.b.y != self.dim.y - 1 and self.a.y != self.dim.y - 1:
                    return self.next()
            elif self.a.y == self.b.y:
                if self.b.x != 0 and self.a.x != 0:
                    return self.next()
                elif self.b.x != self.dim.x - 1 and self.a.x != self.dim.x - 1:
                    return self.next()
        return True

    def nexta(self):
        """passe au segment suivant, return False si on est dernier"""
        if self.bord1 == 1:
            if self.a.y < self.dim.y - 1:
                self.a.y += 1
            else:
                self.bord1 = 2
                self.a.x = self.dim.x - 2
        elif self.bord1 == 2:
            if self.a.x > 0:
                self.a.x -= 1
            else:
                self.a.y = self.dim.y - 2
                self.bord1 = 3
        elif self.bord1 == 3:
            if self.a.y > 0:
                self.a.y -= 1
            else:
                self.a.x = 1
                self.bord1 = 0
        elif self.bord1 == 0:
            if self.a.x < self.dim.x - 1:
                self.a.x += 1
            else:
                self.bord1 = 1
                self.a.y = 1

        if self.bord1 == 0:
            if self.a.x == 0:
                self.b.x, self.b.y = self.dim.x - 1, 0
            else:
                self.b.x, self.b.y = self.dim.x - 1, 1
        elif self.bord1 == 1:
            if self.a.y == 0:
                self.b.x, self.b.y = self.dim.x - 1, self.dim.y - 1
            else:
                self.b.x, self.b.y = self.dim.x - 2, self.dim.y - 1
        elif self.bord1 == 2:
            if self.a.x == self.dim.x - 1:
                self.b.x, self.b.y = 0, self.dim.y - 1
            else:
                self.b.x, self.b.y = 0, self.dim.y - 2
        else:
            if self.a.y == self.dim.y - 1:
                self.b.x, self.b.y = 0, 0
            else:
                self.b.x, self.b.y = 1, 0
        self.bord2 = (self.bord1 + 1) % 4

        if self.a.x == 0 and self.a.y == 0:
            return False
        else:
            return True

    def milieu(self):
        """
        Initialize le segment au milieu de l'image.
        """
        self.bord1, self.bord2 = 2, 0
        self.a.x, self.a.y = 23, self.dim.y - 1
        self.b.x, self.b.y = 23, 0
