
===================
Régression quantile
===================

La régression quantile est moins sensible aux points aberrants.
Elle peut être définie comme une régression avec une norme
*L1* (une valeur absolue).

.. contents::
    :local:

Médiane et valeur absolue
=========================

On considère un ensemble de nombre réels
:math:`\acc{X_1,...,X_n}`. La médiane est le
nombre *M* qui vérifie :

.. math::

    \sum_i \indicatrice{X_i < M} = \sum_i \indicatrice{X_i > M}

Plus simplement, la médiane est obtenue en triant les éléments
:math:`\acc{X_1,...,X_n}` par ordre croissant. La médiane
est alors le nombre au milieu :math:`X_{\cro{\frac{n}{2}}}`.

.. mathdef::
    :title: Médiane et valeur absolue
    :tag: propriété

    La médiane *M* de l'ensemble :math:`\acc{X_1,...,X_n}`
    minimise la quantité :math:`E = \sum_i \abs{X_i - M}`.

Avant de démontrer la propriété, voyons ce qu'il
se passe entre deux réels. La médiane de :math:`\acc{A,B}`
peut être n'importe où sur le segment.

.. image:: images/mediane1.png
    :height: 100

De manière évidente, les distances
des deux côtés du point *M* sont égales :
:math:`a+b = c+d`. Mais si *M* n'est pas sur le segment,
on voit de manière évidente que la somme
des distances sera plus grande.

.. image:: images/mediane2.png
    :height: 100

N'importe quel point sur le segment *M* minimise
:math:`\abs{A - M} + \abs{B - M}`.
On revient aux *n* réels triés par ordre croissant
:math:`\acc{X_1,...,X_n}` et on considère les paires
:math:`(X_1, X_n)`, :math:`(X_2, X_{n-1})`, ...,
:math:`\pa{X_{\cro{\frac{n}{2}}}, X_{\cro{\frac{n}{2}+1}}}`.
L'intersection de tous ces intervalles est
:math:`\pa{X_{\cro{\frac{n}{2}}}, X_{\cro{\frac{n}{2}+1}}}`
et on sait d'après la petit exemple avec deux points
que n'importe quel point dans cet intervalle minimise
:math:`\abs{X_1 - M} + \abs{X_n - M} + \abs{X_2 - M} + \abs{X_{n-1} - M} + ... = E`.
La propriété est démontrée.

Régression quantile
===================

Maintenant que la médiane est définie par un problème
de minimisation, il est possible de l'appliquer à un
problème de régression.

.. mathdef::
    :title: Régression quantile
    :tag: Définition

    On dispose d'un ensemble de *n* couples
    :math:`(X_i, Y_i)` avec :math:`X_i \in \R^d`
    et :math:`Y_i \in \R`. La régression quantile
    consiste à trouver :math:`\alpha, \beta` tels que la
    somme :math:`\sum_i \abs{\alpha + \beta X_i - Y_i}`
    est minimale.

Résolution d'une régression quantile
====================================

La première option consiste à utiliser une méthode
de descente de gradient puisque la fonction
:math:`E = \sum_i \abs{X_i - M}` est presque
partout dérivable. Une autre option consiste à
utiliser l'algorithme
`Iteratively reweighted least squares <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.
L'implémentation est faite par la classe
`QuantileLinearRegression <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/quantile_regression.html#mlinsights.mlmodel.quantile_regression.QuantileLinearRegression>`_.

.. mathdef::
    :title: Iteratively reweighted least squares
    :tag: Algorithme

    On souhaite trouver les paramètres :math:`\Omega`
    qui minimise :

    .. math::

        E = \sum_i \abs{Y_i - f(X_i, \Omega)}

    *Etape 1*

    On pose :math:`\forall i, \, w_i^t = 1`.

    *Etape 2*

    On calcule :math:`\Omega_t = \arg \min E(\Omega)` avec
    :math:`E_t(\Omega) = \sum_i w_i^t \pa{Y_i - f(X_i, \Omega)}^2`.

    *Etape 3*

    On met à jour les poids
    :math:`w_i^{t+1} = \frac{1}{\abs{Y_i - f(X_i, \Omega_t)}}`.
    Puis on retourne à l'étape 2.

Il y a plusieurs choses à démontrer. On suppose que l'algorithme
converge, ce qu'on n'a pas encore démontré. Dans ce cas,
:math:`\Omega_t = \Omega_{t+1}` et les coefficients
:math:`\Omega_t` optimise la quantité :

.. math::

    \sum_i w_i^t \pa{Y_i - f(X_i, \Omega)}^2 =
    \sum_i \frac{\pa{Y_i - f(X_i, \Omega)}^2}{\abs{Y_i - f(X_i, \Omega_t)}} =
    \sum_i \abs{Y_i - f(X_i, \Omega)}

On remarque également que :math:`E_t(\Omega_t}` est l'erreur *L1*
pour les paramètres :math:`\Omega`.
Donc si l'algorithme converge, celui-ci optimise bien
l'erreur de la régression quantile. On va maintenant montrer
que :math:`E_{t+1}(\Omega_{t+2}) \leqslant E_t(\Omega_{t+1})`.
On sait déjà que :math:`E_t{\Omega_{t+1}} \leqslant E_t(\Omega_{t})`
puisque :math:`\Omega_{t+1}` minimise :math:`E_t{\Omega}`.
On calcule :

.. math::
    :nowrap:

    \begin{array}{rcl}
    E_{t+1}(\Omega_{t+1}) - E_t(\Omega_{t+1}) &=&
    \frac{\pa{Y_i - f(X_i, \Omega_{t+1})}^2}{\abs{Y_i - f(X_i, \Omega_{t+1})}} -
    \frac{\pa{Y_i - f(X_i, \Omega_{t+1})}^2}{\abs{Y_i - f(X_i, \Omega_t)}}
    \end{array}

.. toctree::

    ../notebooks/quantile_regression_example
