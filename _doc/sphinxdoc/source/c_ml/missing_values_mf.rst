
===============================================
Valeurs manquantes et Factorisation de Matrices
===============================================

Contourner le problèmes des valeurs manquantes veut souvent dire,
soit supprimer les enregistrements contenant des valeurs manquantes,
soit choisir un modèle capable de faire avec ou soit trouver un moyen de les
remplacer. L'idée de ce chapitre n'est pas d'étudier ces trois cas
(voir `Imputation de données manquantes <https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-app-idm.pdf>`_)
mais plutôt de regarder un algorithme conçu pour autre chose mais dont l'objectif
est de donner une valeur à des données manquantes :
la `factorisation de matrice non négative <https://en.wikipedia.org/wiki/Non-negative_matrix_factorization>`_.
Cette méthode est utilisée dans le cadre de la recommandation de produits
à des utilisateurs.
Lire également [Acara2011]_, [Gupta2010]_.

.. contents::
    :local:

Factorisation de matrice et rang
================================

La `factorisation d'une matrice positive <https://en.wikipedia.org/wiki/Non-negative_matrix_factorization>`_
est un problème d'optimisation qui consiste à trouver pour une matrice
:math:`M \in \mathcal{M}_{pq}` à coefficients positifs ou nuls :

.. math::

    M = WH

Où :math:`W` et :math:`H` sont de rang :math:`k` et de dimension
:math:`W \in \mathcal{M}_{pk}` et :math:`H \in \mathcal{M}_{kq}`.
Si :math:`k < rang(M)`, le produit :math:`WH` ne peut être égal à :math:`M`.
Dans ce cas, on cherchera les matrices qui minimise :

.. mathdef::
    :title: Factorisation de matrices positifs
    :tag: Problème

    Soit :math:`M \in \mathcal{M}_{pq}`, on cherche les matrices à coefficients positifs
    :math:`W \in \mathcal{M}_{pk}` et :math:`H \in \mathcal{M}_{kq}` qui sont solution
    du problème d'optimisation :

    .. math::

        \min_{W,h}\acc{\norme{M-WH}^2} = \min_{W,H} \sum_{ij} (m_{ij} - \sum_k w_{ik} h_{kj})^2

Quelques cas simples
====================

Le notebook :ref:`valeursmanquantesmfrst` montre la décroissante de l'erreur
en fonction du rang et l'impact de la corrélation sur cette même erreur.
Le dernier paragraphe montre qu'il n'existe pas de solution unique à un problème donné.
L'exemple suivant s'intéresse à une matrice 3x3.
Les trois points forment un triangle dans un plan.

.. plot::
    :include-source:

    import numpy
    W = numpy.array([[0.5, 0.5, 0], [0, 0, 1]]).T
    H = numpy.array([[1, 1, 0], [0.0, 0.0, 1.0]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    wh = W @ H
    ax.scatter(M[:,0], M[:,1], M[:,2], c='b', marker='o', s=20)
    ax.scatter(wh[:,0], wh[:,1], wh[:,2], c='r', marker='^')
    plt.show()

On peut voir la matrice :math:`M` comme un ensemble de :math:`n=3` points dans un espace vectoriel.
La matrice :math:`W` est un ensemble de :math:`k < n` points dans le même espace.
La matrice :math:`WH`, de rang :math:`k` est une approximation de cet ensemble
dans le même espace, c'est aussi :math:`n` combinaisons linéaires de :math:`k`
points de façon à former :math:`n` points les plus proches proches de
:math:`n` points de la matrice :math:`M`.

Intuition géométrique
=====================

L'exemple précédente suggère une interprétation géométrique d'une factorisation
de matrice. Sans valeur manquante, ce problème est équivalent à une
`Analyse en Composantes Principales (ACP) <https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales>`_
(voir aussi [Boutsidis2008]_ (décomposition en valeurs singulières comme algorithme d'initialisation).
Nous allons le montrer grâce à un lemme et un théorème.

.. mathdef::
    :title: Factorisation de matrice contraintes
    :tag: Lemme
    :lid: lemme_mf_1

    Soit :math:`M=WH`, les matrices :math:`W` et :math:`H`
    sont de rang :math:`K`. On note :math:`M=(m_{ij})`,
    :math:`W=(w_{ik})`, :math:`H=(h_{kj})`. On suppose que les matrices
    sont solutions du problème d'optimisation
    :math:`\min_{W,H} \norm{ M - WH }^2`.
    Alors on peut trouver deux autres matrices :math:`W'` et :math:`H'`
    telles que:

    .. math::

        \begin{array}{l} \norm{ M - WH }^2 = \norm{ M - W'H' }^2 \\ \forall j, \sum_k h'_{kj} = 1 \end{array}

    On rappelle que les coefficients de la matrice :math:`H` sont positifs.

Avec cette écriture, la matrice :math:`W'H'`
est une façon de former :math:`n` points dans l'enveloppe convexe déterminée par
:math:`k` autres.

Quelques résultats
==================

Bibliographie
=============

.. [Acara2011] Scalable tensorfactorizations for incomplete data,
    *Evrim Acara Daniel, M.Dunlavyc, Tamara G.Koldab. Morten Mørupd*,
    Chemometrics and Intelligent Laboratory Systems,
    Volume 106, Issue 1, 15 March 2011, Pages 41-56,
    `ArXiv <https://arxiv.org/pdf/1005.2197.pdf>`_

.. [Boutsidis2008] SVD-based initialization: A head start for nonnegative matrix factorization.
    *Christos Boutsidis and Efstratios Gallopoulos*
    Pattern Recognition, 41(4): 1350-1362, 2008.

.. [Gupta2010] Additive Non-negative Matrix Factorization for Missing Data,
    Mithun Das Gupta,
   `ArXiv <https://arxiv.org/abs/1007.0380>`_
