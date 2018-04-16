# -*- coding: utf-8 -*-
"""
@file
@brief Détecte les segments dans une image.
"""
import math
import numpy
import copy
import time
import sys
import os
import cv2
from .queue_binom import tabule_queue_binom
from .geometrie import Point, Segment
from .detection_segment_segangle import SegmentBord
from .detection_nfa import LigneGradient, InformationPoint


def detect_segments(image_name, proba_bin=1.0 / 16,
                    cos_angle=math.cos(1.0 / 16 / 2 * (math.pi * 2)),
                    seuil_nfa=1e-5, seuil_norme=2, angle=math.pi / 24.0,
                    stop=-1, verbose=False):
    """
    Détecte les segments dans une image.

    @param  image_name  image file name
    @param  proba_bin   est en fait un secteur angulaire (360 / 16)
                        qui determine la proximite de deux directions
    @param  cos_angle   est le cosinus de l'angle correspondant à ce secteur angulaire
    @param  seuil_nfa   au delà de ce seuil, on considere qu'un segment
                        génère trop de fausses alertes pour être sélectionné
    @param  seuil_norme norme en deça de laquelle un gradient est trop
                        petit pour etre significatif (c'est du bruit)
    @param  angle       lorsqu'on balaye l'image pour détecter les segments,
                        on tourne en rond selon les angles 0, angle, 2*angle,
                        3*angle, ...
    @param  stop        arrête après avoir collecté tant de segments
    @param  verbose     affiche l'avancement
    @return             les segments
    """
    img = cv2.imread(image_name)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # on calcule les tables de la binomiale pour eviter d'avoir a le fait a
    # chaque fois qu'on en a besoin
    xx, yy = img.shape[:2]
    nbbin = int(math.ceil(math.sqrt(xx * xx + yy * yy)))
    binomiale = tabule_queue_binom(nbbin, proba_bin)

    # transformation du gradient
    grad = {}
    for i in range(0, xx):
        for j in range(0, yy):
            grad[i, j] = Point(sobelx[i, j], sobely[i, j])

    # nb_seg est le nombre total de segment de l'image
    # il y a xx * yy pixels possibles dont (xx*yy)^2 couples de pixels (donc de segments)
    nb_seg = xx * xx * yy * yy

    # couleurs possibles pour afficher des segments
    couleur = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # on cree une instance de la classe permettant de parcourir
    # tous les segments de l'image reliant deux points du contour
    seg = SegmentBord(Point(xx, yy))

    # initialisation avant de parcourir l'image
    segment = []        # resultat, ensemble des segments significatifs
    ti = time.clock()  # memorise l'heure de depart
    # pour savoir combien de segments on a deja visite (seg)
    n = 0
    cont = True         # condition d'arret de la boucle

    # on cree une classe permettant de recevoir les informations relatives
    # a l'image et au gradient pour un segment reliant deux points
    # du contour de l'image
    points = [InformationPoint(Point(0, 0), False, 0)
              for i in range(0, xx + yy)]
    ligne = LigneGradient(points, seuil_norme=seuil_norme, seuil_nfa=seuil_nfa)

    # premier segment
    seg.premier()

    # autres variables a decouvrir en cours de route
    clast = 0
    nblast = 0

    # tant qu'on a pas fini
    while cont:

        # calcule les informations relative a un segment de l'image reliant deux bords
        # position des pixels, norme du gradient, alignement avec le segment
        seg.decoupe_gradient(grad, cos_angle, ligne, seuil_norme)

        if len(ligne) > 3:
            # si le segment contient plus de trois pixels
            # alors on peut se demander s'il inclut des sous-segments significatifs
            res = ligne.segments_significatifs(binomiale, nb_seg)

            # on ajoute les resultats a la liste
            segment.extend(res)
            if stop > 0 and len(segment) >= stop:
                break

        # on passe au segment suivant
        cont = seg.next()
        n += 1

        # pour verifier que cela avance
        if verbose and n % 1000 == 0:
            print("n = ", n, " ... ", len(segment), " temps ",
                  "%2.2f" % (time.clock() - ti), " sec")

    return segment
