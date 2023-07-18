# -*- coding: utf-8 -*-
"""
@file
@brief Détecte les segments dans une image.
"""
import math
import copy
import time
import numpy
from PIL import Image, ImageDraw
from .queue_binom import tabule_queue_binom
from .geometrie import Point
from .detection_segment_segangle import SegmentBord
from .detection_nfa import LigneGradient, InformationPoint


def convert_array2PIL(img, mode=None):
    """
    Convertit une image donnée sous la forme d'un array
    au format :epkg:`numpy:array`.

    @param      img     :epkg:`numpy:array`
    @param      mode    voir `modes <https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#modes>`_,
                        si None, essaye de deviner.
    @return             *PIL*

    Le mode ``'binary'`` convertit une image issue
    de la fonction @see fn random_noise_image.
    """
    if mode == "binary":
        fimg = img.astype(numpy.float32)
        img255 = (-fimg + 1) * 255
        img = img255.astype(numpy.uint8)
        mode = None
    return _load_image(img, "PIL", mode=mode)


def convert_PIL2array(img):
    """
    Convertit une image donnée sous la forme d'une image :epkg:`Pillow`
    au format :epkg:`numpy:array`.

    @param      img     :epkg:`Pillow`
    @return             :epkg:`numpy:array`
    """
    return _load_image(img, "array")


def _load_image(img, format="PIL", mode=None):
    """
    Charge une image en différents formats.

    @param      img     image (*array*, *PIL*, filename)
    @param      format  *array* ou *PIL*
    @param      mode    voir `modes <https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#modes>`_,
                        si None, essaye de deviner.
    @return             *PIL*
    """
    if isinstance(img, str):
        img = Image.open(img)
        return _load_image(img, format)
    if isinstance(img, Image.Image):
        if format == "PIL":
            return img
        if format == "array":
            d1, d0 = img.size[1], img.size[0]
            img = numpy.array(img.getdata(), dtype=numpy.uint8)
            if len(img.shape) == 1:
                gray = img.shape[0] - d1 * d0
            elif len(img.shape) == 2:
                gray = img.shape[0] * img.shape[1] - d1 * d0
            elif len(img.shape) == 3:
                gray = img.shape[0] * img.shape[1] * img.shape[2] - d1 * d0
            else:
                raise ValueError(f"Unexpected shape {img.shape}")  # pragma: no cover
            if gray == 0:
                img = img.reshape((d1, d0))
            else:
                img = img.reshape((d1, d0, 3))
            return img
        raise ValueError(f"Unexpected value for fomat: '{format}'")  # pragma: no cover
    if isinstance(img, numpy.ndarray):
        if format == "array":
            return img
        if format == "PIL":
            return Image.fromarray(img, mode=mode)
        raise ValueError(f"Unexpected value for fomat: '{format}'")  # pragma: no cover
    raise TypeError(f"numpy array expected not {type(img)}")  # pragma: no cover


def compute_gradient(img, color=None):
    """
    Retourne le gradient d'une image sous forme d'une matrice
    de Point, consideres ici comme des vecteurs.
    """
    return _calcule_gradient(img, color=color)


def _calcule_gradient(img, color=None):
    """
    Retourne le gradient d'une image sous forme d'une matrice
    de Point, consideres ici comme des vecteurs.

    @param      img     *fichier*, *array*, *PIL* (image en niveau de gris)
    @param      method  ``'fast'`` or not
    @param      color   calcule le gradient pour cette couleur, None
                        si l'image est en niveau de gris
    @return             array of *shape (y, x, 2)*, first dimension is *dx*,
                        second one is *dy*
    """
    img = _load_image(img, "array")
    img = img.astype(numpy.float32)
    if color is not None:
        img = img[:, :, color]

    dx1 = img[:, 1:-1] - img[:, :-2]
    dx2 = img[:, 2:] - img[:, 1:-1]
    dx = (dx1 + dx2) / 2

    dy1 = img[1:-1, :] - img[:-2, :]
    dy2 = img[2:, :] - img[1:-1, :]
    dy = (dy1 + dy2) / 2
    res = numpy.zeros(img.shape + (2,))
    res[:, 1:-1, 0] = dx
    res[1:-1, :, 1] = dy
    return res


def plot_gradient(image, gradient, more=None, direction=-1):
    """
    Construit une image a partir de la matrice de gradient
    afin de pouvoir l'afficher grace au module pygame,
    cette fonction place directement le resultat dans image,
    si direction > 0, cette fonction affiche egalement le gradient sur
    l'image tous les 10 pixels si direction vaut 10.
    """
    image_ = _load_image(image, "PIL")
    image = ImageDraw.Draw(image_)
    X, Y = image_.size
    if direction != -1:
        for x in range(0, X - 1):
            for y in range(0, Y - 1):
                n = gradient[y, x]
                if more is None:
                    v = int((n[0] ** 2 + n[1] ** 2) ** 0.5 + 0.5)
                elif more == "x":
                    v = int(n[0] / 2 + 127 + 0.5)
                else:
                    v = int(n[1] / 2 + 127 + 0.5)
                image.line([(x, y), (x, y)], fill=(v, v, v))
    if direction in (0, -1):
        pass
    elif direction > 0:
        # on dessine des petits gradients dans l'image
        for x in range(0, X, direction):
            for y in range(0, Y, direction):
                n = gradient[y, x]
                t = (n[0] ** 2 + n[1] ** 2) ** 0.5
                if t == 0:
                    continue
                m = copy.copy(n)
                m /= t
                if t > direction:
                    t = direction
                if t < 2:
                    t = 2
                m *= t
                image.line([(x, y), (x + int(m[0]), y + int(m[1]))], fill=(255, 255, 0))
    elif direction == -2:
        # derniere solution, la couleur represente l'orientation
        # en chaque point de l'image
        for x in range(0, X):
            for y in range(0, Y):
                n = gradient[y, x]
                i = int(-n[0] * 10 + 128)
                j = int(n[1] * 10 + 128)
                i, j = min(i, 255), min(j, 255)
                i, j = max(i, 0), max(j, 0)
                image.line([(x, y), (x, y)], fill=(0, j, i))
    else:
        raise ValueError(  # pragma: no cover
            f"Unexpected value for direction={direction}"
        )

    return image_


def plot_segments(image, segments, outfile=None, color=(255, 0, 0)):
    """
    Dessine les segments produits par la fonction
    @see fn detect_segments

    @param  image       image (*fichier*, *array*, *PIL*)
    @param  segments    résultats de la fonction @see fn detect_segments
    @param  outfile     fichier de sortie
    @param  color       couleur
    @return             nom de fichier ou image
    """
    image = _load_image(image, "PIL")
    draw = ImageDraw.Draw(image)
    for seg in segments:
        draw.line([(seg.a.x, seg.a.y), (seg.b.x, seg.b.y)], fill=color)
    if outfile is not None:
        image.save(outfile)
        return outfile
    return image


def detect_segments(
    image,
    proba_bin=1.0 / 16,
    cos_angle=math.cos(1.0 / 16 / 2 * (math.pi * 2)),
    seuil_nfa=1e-5,
    seuil_norme=2,
    angle=math.pi / 24.0,
    stop=-1,
    verbose=False,
):
    """
    Détecte les segments dans une image.

    :param image: image (*fichier*, *array*, *PIL*)
    :param proba_bin: est en fait un secteur angulaire (360 / 16)
        qui determine la proximite de deux directions
    :param cos_angle: est le cosinus de l'angle correspondant à ce secteur angulaire
    :param seuil_nfa: au delà de ce seuil, on considere qu'un segment
        génère trop de fausses alertes pour être sélectionné
    :param seuil_norme: norme en deça de laquelle un gradient est trop
        petit pour etre significatif (c'est du bruit)
    :param angle: lorsqu'on balaye l'image pour détecter les segments,
        on tourne en rond selon les angles 0, angle, 2*angle, 3*angle, ...
    :param stop: arrête après avoir collecté tant de segments
    :param verbose: affiche l'avancement
    :return: les segments
    """
    gray_image = _load_image(image, "PIL").convert("L")
    grad = _calcule_gradient(gray_image)

    # on calcule les tables de la binomiale pour eviter d'avoir a le fait a
    # chaque fois qu'on en a besoin
    yy, xx = grad.shape[:2]
    nbbin = int(math.ceil(math.sqrt(xx * xx + yy * yy)))
    binomiale = tabule_queue_binom(nbbin, proba_bin)

    # nb_seg est le nombre total de segment de l'image
    # il y a xx * yy pixels possibles dont (xx*yy)^2
    # couples de pixels (donc de segments)
    nb_seg = xx * xx * yy * yy

    # on cree une instance de la classe permettant de parcourir
    # tous les segments de l'image reliant deux points du contour
    seg = SegmentBord(Point(xx, yy))

    # initialisation avant de parcourir l'image
    segment = []  # resultat, ensemble des segments significatifs
    ti = time.perf_counter()  # memorise l'heure de depart
    # pour savoir combien de segments on a deja visite (seg)
    n = 0
    cont = True  # condition d'arret de la boucle

    # on cree une classe permettant de recevoir les informations relatives
    # a l'image et au gradient pour un segment reliant deux points
    # du contour de l'image
    points = [InformationPoint(Point(0, 0), False, 0) for i in range(0, xx + yy)]
    ligne = LigneGradient(points, seuil_norme=seuil_norme, seuil_nfa=seuil_nfa)

    # premier segment
    seg.premier()

    # autres variables a decouvrir en cours de route
    not_aligned = 0

    # tant qu'on a pas fini
    while cont:
        # calcule les informations relative a un segment de l'image reliant deux bords
        # position des pixels, norme du gradient, alignement avec le segment
        seg.decoupe_gradient(grad, cos_angle, ligne, seuil_norme)

        if len(ligne) > 3 and ligne.has_aligned_point():
            # si le segment contient plus de trois pixels
            # alors on peut se demander s'il inclut des sous-segments significatifs
            res = ligne.segments_significatifs(binomiale, nb_seg)

            # on ajoute les resultats à la liste
            segment.extend(res)
            if len(segment) >= stop > 0:
                break
        else:
            not_aligned += 1

        # on passe au segment suivant
        cont = seg.next()  # pylint: disable=E1102
        n += 1

        # pour verifier que cela avance
        if verbose and n % 1000 == 0:
            print(  # pragma: no cover
                "n = ",
                n,
                " ... ",
                len(segment),
                " temps ",
                f"{time.perf_counter() - ti:2.2f}",
                " sec",
                "nalign",
                not_aligned,
            )

    return segment
