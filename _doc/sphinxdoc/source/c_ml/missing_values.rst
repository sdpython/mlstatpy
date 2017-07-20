
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

Quelques cas simples
====================

Aucune corrélation
++++++++++++++++++

Rang et corrélation
+++++++++++++++++++

Intuition géométrique
=====================

Quelques résultats
==================

Bibliographie
=============

.. [Acara2011] Scalable tensorfactorizations for incomplete data,
    *Evrim Acara Daniel, M.Dunlavyc, Tamara G.Koldab. Morten Mørupd*,
    Chemometrics and Intelligent Laboratory Systems,
    Volume 106, Issue 1, 15 March 2011, Pages 41-56,
    `ArXiv <https://arxiv.org/pdf/1005.2197.pdf>`_

.. [Gupta2010] Additive Non-negative Matrix Factorization for Missing Data,
    Mithun Das Gupta,
   `ArXiv <https://arxiv.org/abs/1007.0380>`_
