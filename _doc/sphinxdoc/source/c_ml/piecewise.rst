
================================
Régression linéaire par morceaux
================================

Le paragraphe :ref:`regressionlineairerst`
étudie le lien entre le coefficient :math:`R^2`
et la corrélation pour finalement illustrer
une façon de réaliser une régression linéaire par
morceaux. L'algorithme s'appuie sur un arbre
de régression pour découper en morceaux ce qui
n'est pas le plus satisfaisant car l'arbre
cherche à découper en segment en approximant
la variable à régresser *Y* par une constante sur chaque
morceaux et non une droite.
On peut se poser la question de comment faire
pour construire à un algorithme qui découpe en approximant
*Y* par une droite et non une constante. Le plus dur
n'est pas de le faire mais de le faire efficacement.
Et pour comprendre là où je veux vous emmener, il faudra
un peu de mathématiques.

.. contents::
    :local:
    
Le problème
===========

Tout d'abord, une petite
illustration du problème avec la classe
`PiecewiseRegression <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression.html>`_
implémentée selon l'API de :epkg:`scikit-learn`.

.. toctree::
    :maxdepth: 1
    
    ../notebooks/piecewise_linear_simple


Régression linéaire et corrélation
==================================

Idée de l'algorithme
====================

Implémentation
==============

