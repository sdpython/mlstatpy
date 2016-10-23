#-*- coding: utf-8 -*-
"""
@file
@brief Ce module construit les probabilités d'une loi binomiale :math:`B(n,p)`.
"""


def tabule_queue_binom(n, p):
    """
    Retourne un dictionnaire dont la clé est couple d'entiers *(a,b)*
    si *t* est le resultat, alors :math:`t=[(a,b)]` est la probabilité
    qu'il y ait *b* événements parmi *a* sachant que la probabilité d'un
    événement est *p* : :math:`t [ (a,b) ] = C_a^b p^b (1-p)^ {(a-b)}`

    Pour aller plus vite, ces probabilités sont estimées par récurrence :

    * :math:`\\forall m, \\; t [(m,0)] = 1.0`
    * :math:`\\forall m, \\; t [(m,m+1)] = 0.0`
      et :math:`t[(m,k)] = p * t [ (m-1, k-1)] + (1-p) * t [ (m-1,k) ]`

    Cette fonction calcule tous les coefficients :math:`t [ (a,b) ]` pour une
    probabilité :math:`p` donnée et :math:`b \\infegal a \\infegal n`.

    Ces probabilités sont stockées dans un dictionnaire car s'ils étaient
    stockées dans une matrice, celle-ci serait triangulaire inférieure.
    """
    t = {}
    t[(0, 0)] = 1.0
    t[(0, 1)] = 0.0
    for m in range(1, n + 1):
        t[(m, 0)] = 1.0
        t[(m, m + 1)] = 0.0
        for k in range(1, m + 1):
            t[(m, k)] = p * t[(m - 1, k - 1)] + (1 - p) * t[(m - 1, k)]
    return t
