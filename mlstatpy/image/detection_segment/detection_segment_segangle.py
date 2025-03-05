import math
import copy
from .detection_segment_bord import SegmentBord_Commun
from .geometrie import Point


class SegmentBord(SegmentBord_Commun):
    """
    Définit un segment allant d'un bord à un autre de l'image,
    la classe va balayer toutes les orientations possibles,
    pour chaque orientation, elle va ensuite balayer toute l'image,

    * **angle** : orientation de balayage
    * **fin** : pour le balayage de l'image à une orientation donnee,
      le segment part d'un bord, fin désigne le dernier pixel
      du contour à envisager avec cette orientation
    * **vecteur** : vecteur directeur correspondant à angle
    * **bord1** : numero du bord (0,1,2,3) correspondant à ``self.a``,
      0 bord droit, 1 bord haut, 2 bord gauche, 3 bord bas
    * **dangle** : orientation à visiter 0, dangle, 2*dangle, 3*dangle, ...

    Pour parcourir les segments de l'image, on part d'un premier segment
    (methode premier), segment horizontal, bord 0,
    la méthode next passe au segment suivant jusqu'au dernier auquel
    cas la methode retourne *False*.
    Les segments sont orientés, si un gradient est proche du vecteur
    normal, l'opposé ne l'est pas.
    """

    # voir la remarque de la classe Point a propos de __slots__
    __slots__ = ("angle", "fin", "vecteur", "bord1", "dangle")  # noqa: RUF023

    def __init__(self, dim, dangle=math.pi / 24.0):
        """initialise les dimensions et
        fait sorte que la classe contienne le premier segment"""
        SegmentBord_Commun.__init__(self, dim)
        self.premier()
        self.dangle = dangle

    def __str__(self):
        """permet d'afficher le segment"""
        s = SegmentBord_Commun.__str__(self)
        s += " -- bord " + str(self.bord1)
        s += " -- fin " + str(self.fin)
        s += " -- a " + f"{self.angle * 180 / math.pi:3.1f}"
        s += " -- vec " + f"{self.vecteur.x:2.2f},{self.vecteur.y:2.2f}"
        return s

    def premier(self):
        """définit le premier segment, horizontal, part du bord gauche"""
        self.angle = 0
        self.fin = Point(0, 0)
        self.calcul_vecteur()

    def milieu(self):
        """
        Un autre segment, pour débugger le programme,
        choisit une orientation pour laquelle on sait que le
        résultat doit être un segment significatif,
        la methode *next* ira plus vite au dernier segment.
        """
        self.angle = math.pi / 2
        self.calcul_vecteur()

    def calcul_vecteur(self):
        """
        En fonction de l'angle, calcule le vecteur direction du segment,
        ensuite fixe la première extrémité du segment ``self.a``
        et détermine la dernière des premières extremités pour
        un segment de cette orientation (angle).
        """
        self.vecteur = Point(math.cos(self.angle), math.sin(self.angle))

        # le vecteur est horizontal
        if abs(self.vecteur.x) < 1e-5:
            if self.vecteur.y > 0:
                # vers le haut, un seul bord pour la premiere extremite
                self.bord1 = 3
                self.a.x = 0
                self.a.y = 0
                self.fin.x = self.dim.x - 1
                self.fin.y = 0
                return self.calcul_vecteur_fin()
            else:
                # vers le bas, un seul bord pour la premiere extremite
                self.bord1 = 1
                self.a.x = self.dim.x - 1
                self.a.y = self.dim.y - 1
                self.fin.x = 0
                self.fin.y = self.dim.y - 1
                return self.calcul_vecteur_fin()

        # le vecteur est vertical
        if abs(self.vecteur.y) < 1e-5:
            if self.vecteur.x < 0:
                # vers la droite, un seul bord pour la premiere extremite
                self.bord1 = 0
                self.a.x = self.dim.x - 1
                self.a.y = 0
                self.fin.x = self.dim.x - 1
                self.fin.y = self.dim.y - 1
                return self.calcul_vecteur_fin()
            else:
                # vers la gauche, un seul bord pour la premiere extremite
                self.bord1 = 2
                self.a.x = 0
                self.a.y = self.dim.y - 1
                self.fin.x = 0
                self.fin.y = 0
                return self.calcul_vecteur_fin()

        if self.vecteur.x < 0:
            if self.vecteur.y < 0:
                # en bas a gauche, deux bords pour la premiere extremite
                self.bord1 = 0
                self.a.x = self.dim.x - 1
                self.a.y = 0
                self.fin.x = 0
                self.fin.y = self.dim.y - 1
                return self.calcul_vecteur_fin()
            else:
                # en haut a gauche, deux bords pour la premiere extremite
                self.bord1 = 3
                self.a.x = 0
                self.a.y = 0
                self.fin.x = self.dim.x - 1
                self.fin.y = self.dim.y - 1
                return self.calcul_vecteur_fin()
        else:
            if self.vecteur.y < 0:
                # en haut a droite, deux bords pour la premiere extremite
                self.bord1 = 1
                self.a.x = self.dim.x - 1
                self.a.y = self.dim.y - 1
                self.fin.x = 0
                self.fin.y = 0
                return self.calcul_vecteur_fin()
            else:
                # en bas a droite, deux bords pour la premiere extremite
                self.bord1 = 2
                self.a.x = 0
                self.a.y = self.dim.y - 1
                self.fin.x = self.dim.x - 1
                self.fin.y = 0
                return self.calcul_vecteur_fin()

    def calcul_vecteur_fin(self):
        """propose une seconde extrémité connaissant la première,
        beaucoup plus loin en conservant la meme orientation,
        du moment qu'on traverse l'image"""
        t = self.dim.x + self.dim.y
        self.b.x = int(self.a.x + self.vecteur.x * t)
        self.b.y = int(self.a.y + self.vecteur.y * t)

    def directeur(self):
        """retourne une copie du vecteur directeur"""
        return copy.copy(self.vecteur)

    def next(self):
        """passe au segment suivant dans le parcours de l'image"""

        if self.angle >= math.pi * 2 - 1e-5:
            # toute orientation visitee
            return False

        if self.a == self.fin:
            # tout vecteur visitee pour la meme orientation,
            # on passe a l'orientation suivante
            self.angle += self.dangle
            self.calcul_vecteur()
            if self.angle >= math.pi * 2:
                return False
            else:
                return True
        else:
            # on passe au segment suivant selon la meme orientation,
            # tout depend du bord sur lequel on est
            if self.bord1 == 0:
                # bord droit
                if self.a.y < self.dim.y - 1:
                    # pas besoin de changer de bord
                    self.a.y += 1
                else:
                    # on passe au bord suivant, bord haut
                    self.bord1 = 1
                    self.a.x -= 1
            elif self.bord1 == 1:
                # bord haut, meme raisonnement que pour le premier bord
                if self.a.x > 0:
                    self.a.x -= 1
                else:
                    self.bord1 = 2
                    self.a.y -= 1
            elif self.bord1 == 2:
                # bord gauche, meme raisonnement que pour le premier bord
                if self.a.y > 0:
                    self.a.y -= 1
                else:
                    self.bord1 = 3
                    self.a.x += 1
            elif self.bord1 == 3:
                # bord bas, meme raisonnement que pour le premier bord
                if self.a.x < self.dim.x - 1:
                    self.a.x += 1
                else:
                    self.bord1 = 0
                    self.a.y += 1
            # choisit une derniere extremite
            self.calcul_vecteur_fin()
            return True

    def calcul_bord2(self):
        """calcule précisément la second extrémité, parcourt la demi-droite
        jusqu'à sortir de l'image, le dernier point est la seconde extrémité"""
        a = self.a.arrondi()
        p = copy.copy(self.a)
        n = self.directeur()

        i = 0
        while a.x >= 0 and a.y >= 0 and a.x < self.dim.x and a.y < self.dim.y:
            p += n
            a = p.arrondi()
            i += 1

        self.b.x = a.x
        self.b.y = a.y
        r = -1
        if self.b.x < 0:
            self.b.x = 0
            r = 2
        if self.b.x >= self.dim.x:
            self.b.x = self.dim.x - 1
            r = 0
        if self.b.y < 0:
            self.b.y = 0
            r = 3
        if self.b.y >= self.dim.y:
            self.b.y = self.dim.y - 1
            r = 1
        return r
