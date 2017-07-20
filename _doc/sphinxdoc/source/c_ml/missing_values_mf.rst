
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

Intuition géométrique
=====================

Sans valeur manquante, ce problème est équivalent à une
`Analyse en Composantes Principales (ACP) <https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales>`_.
Voir aussi [Boutsidis2008]_ (décomposition en valeurs singulières comme algorithme d'initialisation.

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
