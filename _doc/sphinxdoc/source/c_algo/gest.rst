
.. _l-k-algo-gest:

=====================
Détection de segments
=====================

.. contents::
    :local:

L'idée
======

Une image aléatoire ressemble à la mire en un temps
où la télévision ne rediffusait pas les programmes diurne
la nuit.

.. image:: bruit.png

Dans ce brouillard aléatoire, la probabilité d'avoir des
points alignés est très faible, si faible que le simple
fait d'en voir est un événement extraordinaire. Trois points
alignés ne sont pas rares, quatre un peu plus, cinq encore plus.
A partir d'un certain seuil, on peut considérer que trop de
points alignés forme une droite et un événement trop rare
pour être ignoré. On cherche à détecter les arêtes dans une
image comme la suivante.

.. image:: gradient--1.png

On calcule le gradient d'une image en noir et blanc.

.. image:: gradient-0.png

Puis on extrait les segments en les considérant comme
des anomalies par rapport à un champ de pixels aléatoire.

.. image:: seg.png

Illustration
============

.. toctree::
    :maxdepth: 1

    ../notebooks/segment_detection

La fonction
:func:`detect_segments <mlstatpy.image.detection_segment.detection_segment.detect_segments>`
lance la détection des segments.

Explications
============

La présentation
`Détection des images dans les images digitales <https://github.com/sdpython/mlstatpy/blob/master/_todo/segment_detection/presentation.pdf>`_
détaille le principe de l'algorithme. L'idée de l'algorithme est assez
proche de la `transformée de Hough <https://fr.wikipedia.org/wiki/Transform%C3%A9e_de_Hough>`_.
Celle-ci est implémentée dans le module
`scikit-image <http://scikit-image.org/docs/dev/api/skimage.transform.html>`_ ou
`opencv <https://docs.opencv.org/2.4.13.6/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html?highlight=hough>`_.

Bibliographie
+++++++++++++

* `From Gestalt Theory to Image Analysis <http://www.math-info.univ-paris5.fr/~moisan/papers/2006-9.pdf>`_,
  Agnès Desolneux, Lionel Moisan, Jean-Michel Morel
* `Learning Equivariant Functions with Matrix Valued Kernels <http://www.jmlr.org/papers/volume8/reisert07a/reisert07a.pdf>`_
* `The Hough Transform Estimator <https://arxiv.org/pdf/math/0503668.pdf>`_
* `An Extension to Hough Transform Based on Gradient Orientation <https://arxiv.org/ftp/arxiv/papers/1510/1510.04863.pdf>`_
