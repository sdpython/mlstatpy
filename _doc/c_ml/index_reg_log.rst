
.. _l-reglog-variations:

#####################
Régression logistique
#####################

La `régression logistique <https://fr.wikipedia.org/wiki/R%C3%A9gression_logistique>`_
est le modèle prédictif le plus simple et celui qu'on préfère
quand il marche car il est facilement interprétable à l'inverse
des modèles non linéaires qui gardent leurs secrets si on s'en tient
seulement à leurs coefficients. Concrètement, on dispose d'un nuage
de point :math:`(X_i, y_i)` où :math:`X_i \in \R^d` est un vecteur
de dimension *d* et :math:`y_i \in \acc{0, 1}` un entier binaire.
Le problème de la régression linéaire consiste à
construire une fonction prédictive
:math:`\hat{y_i} = f(X_i) = <X_i, \beta> = X_i \beta` où
:math:`\beta` est un vecteur de dimension *d*
(voir `classification
<http://www.xavierdupre.fr/app/papierstat/helpsphinx/lectures/regclass.html#classification>`_).
Le signe de la fonction :math:`f(X_i)`
indique la classe de l'observation :math:`X_i` et la valeur
:math:`\frac{1}{1 + e^{f(X)}}` la probabilité d'être dans la classe 1.

.. toctree::
    :maxdepth: 1

    lr_voronoi
    lr_trees
    ../notebooks/reseau_neurones
    survival_analysis
