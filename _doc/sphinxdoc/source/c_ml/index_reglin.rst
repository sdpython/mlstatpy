
##########################################
Régressions linéaires et autres variations
##########################################

La `régression linéaire <https://fr.wikipedia.org/wiki/R%C3%A9gression_lin%C3%A9aire>`_
est le modèle prédictif le plus simple et celui qu'on préfère
quand il marche car il est facilement interprétable à l'inverse
des modèles non linéaires qui gardent leurs secrets si on s'en tient
seulement à leurs coefficiens. Concrètement, on dispose d'un nuage
de point :math:`(X_i, y_i)` où :math:`X_i \in \R^d` est un vecteur
de dimension *d* et :math:`y_i \in \R` un réel. La régression
linéaire consiste à construire une fonction prédictive
:math:`\hat{y_i} = f(X_i) = <X_i, \beta> = X_i \beta` où
:math:`\beta` est un vecteur de dimension *d*. Dans le cas le plus
courant, on modélise les données de telle sorte que :
:math:`y_i = X_i \beta + \epsilon_i` où :math:`\epsilon_i`
suit une loi normale de moyenne nulle et de variance :math:`\sigma`.
Sous cette hypothèse, il 'agit de trouver le vecteur :math:`\beta`
qui minimise la vraisemblance du modèle, ce qui revient à résoudre
le problème d'optimisation :

.. math::

    \min_\beta \sum_i (y_i - X_i \beta)^2

En dérivant, on sait exprimer explicitement la solution.
On note :math:`X = (X_1, ..., X_i, ...)` la matrice où chaque ligne
est une observation :math:`X_i` et :math:`y = (y_1, ..., y_i, ...)`.
:math:`X'` est la transposée de *X*. Alors :

.. math::

    \beta_* = (X'X)^{-1}X'y

Les chapitres suivants explorent d'autres aspects de ce problèmes
comme la régression quantile, la régression linéaire par morceaux,
ou encore l'expression de :math:`\beta` sans calculer de matrice inverse
ni de valeurs propres.

.. toctree::
    :maxdepth: 1

    ../notebooks/regression_lineaire
    ../notebooks/quantile_regression
    regression_quantile
    piecewise
