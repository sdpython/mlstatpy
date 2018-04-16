# -*- coding: utf-8 -*-
"""
@file
@brief Ce module contient la liste des parametres utilises
pour la detection des lignes dans une image.
"""
import math


class NFADefaultParameter:
    """
    Ensemble des paramètres utilisés pour la détection des lignes dans une image.

    * **proba_bin** : est en fait un secteur angulaire (360 / 16)
      qui determine la proximite de deux directions
    * **cos_angle** : est le cosinus de l'angle correspondant à ce secteur angulaire
    * **seuil_nfa** : au delà de ce seuil, on considere qu'un segment
      génère trop de fausses alertes pour être sélectionné
    * **seuil_norme** : norme en deça de laquelle un gradient est trop
      petit pour etre significatif (c'est du bruit)
    * **angle** : lorsqu'on balaye l'image pour détecter les segments,
      on tourne en rond selon les angles 0, angle, 2*angle, 3*angle, ...
    """
    proba_bin = 1.0 / 16
    cos_angle = math.cos(1.0 / 16 / 2 * (math.pi * 2))
    seuil_nfa = 1e-5
    seuil_norme = 2
    angle = math.pi / 24.0
