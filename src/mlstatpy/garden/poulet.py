# -*- coding: utf-8 -*-
"""
@file
@brief Functions for :ref:`l-exemple_optim_alea`.
"""
import math
import random


def factorielle(x):
    """
    Calcule :math:`x!` de façon récursive.
    """
    if x == 0:
        return 1
    else:
        return x * factorielle(x - 1)


def profit(N, X, p, q, s):
    """
    Calcule le profit.

    @param      N       nombre de poulets vendus
    @param      X       nombre de poulets achetés
    @param      p       prix d'achat
    @param      q       prix de vente
    @param      s       prix soldé
    @return             profit
    """
    if X <= N:
        return X * (q - p)
    else:
        return X * (s - p) + N * (q - s)


def proba_poisson(l, i):
    """
    Calcule la probabilité :math:`\\pr{X=i}``
    lorsque :math:`X` suit une loi de Poisson de paramètre
    :math:`\\lambda`.
    """
    return math.exp(-l) * (l ** i) / factorielle(i)


def esperance(X, p, q, s, l):
    """
    Espérance du profit en faisant varier
    le nombre de poulet vendus.

    @param      X       nombre de poulets achetés
    @param      p       prix d'achat
    @param      q       prix de vente
    @param      s       prix soldé
    @param      l       paramètre :math:`\\lambda`
    @return             espérance du profit
    """
    res = 0.0
    for i in range(0, l * 2):
        res += profit(float(i), X, p, q, s) * proba_poisson(l, i)
    return res


def maximum(p, q, s, l):
    """
    Calcule les espérances de profit pour différents nombres
    de poulets achetés.

    @param      p       prix d'achat
    @param      q       prix de vente
    @param      s       prix soldé
    @param      l       paramètre :math:`\\lambda`
    @return             liste ``(X, profit)``
    """
    res = []
    for X in range(0, 2 * l):
        r = esperance(X, p, q, s, l)
        res.append((X, r))
    return res


def find_maximum(res):
    """
    Trouver le couple (nombre de poulets achetés, profit)
    lorsque le profit est maximum.

    @param      res         résultat de la fonction :func:`maximum <mlstatpy.garden.poulet.maximum>`
    @return                 ``(X, profit)`` maximum
    """
    m = (0, 0)
    for r in res:
        if r[1] > m[1]:
            m = r
    return m


def exponentielle(l):
    """
    Simule une loi exponentielle de paramètre :math:`\\lambda`.
    """
    u = random.random()
    return - 1.0 / l * math.log(1.0 - u)


def poisson(l):
    """
    Simule une loi de Poisson de paramètre :math:`\\lambda`.
    """
    s = 0
    i = 0
    while s <= 1:
        s += exponentielle(l)
        i += 1
    return i - 1


def poisson_melange(params, coef):
    """
    Simule une variable selon un mélange de loi de Poisson.

    @param      params      liste de paramètre :math:`\\lambda`
    @param      coef        ``coef[i]`` coefficient associé à la loi de paramètre ``params[i]``
    @return                 valeur simulée
    """
    s = 0
    for i in range(0, len(params)):
        p = poisson(params[i])
        s += p * coef[i]
    return s


def histogramme_poisson_melange(params, coef, n=100000):
    """
    Calcule un histogramme d'un mélange de loi de Poisson.

    @param      params      liste de paramètre :math:`\\lambda`
    @param      coef        ``coef[i]`` coefficient associé à la loi de paramètre ``params[i]``
    @return                 histogramme
    """
    h = [0.0 for i in range(0, 4 * max(params))]
    for i in range(0, n):
        x = poisson_melange(params, coef)
        if x < len(h):
            h[x] += 1
    s = sum(h)
    for i in range(0, len(h)):
        h[i] = float(h[i]) / s
    return h


def f_proba_poisson_melange():
    """
    Wraps function *proba_poisson_melange* to avoid
    global variable.
    """

    proba_poisson_melange_tableau = []

    def proba_poisson_melange(params, coef, i,
                              proba_poisson_melange_tableau=proba_poisson_melange_tableau):
        """
        Calcule la probabilité :math:`\\pr{X=i}``
        lorsque :math:`X` suit un mélange de lois.

        @param      params      liste de paramètre :math:`\\lambda`
        @param      coef        ``coef[i]`` coefficient associé à la loi de paramètre ``params[i]``
        @return                 valeur
        """
        if len(proba_poisson_melange_tableau) == 0:
            proba_poisson_melange_tableau.extend(
                histogramme_poisson_melange(params, coef))
        if i >= len(proba_poisson_melange_tableau):
            return 0.0
        else:
            return proba_poisson_melange_tableau[i]

    return proba_poisson_melange


proba_poisson_melange = f_proba_poisson_melange()
