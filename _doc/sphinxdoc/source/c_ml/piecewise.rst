
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

Problème et regréssion linéaire dans un espace à une dimension
==============================================================

Tout d'abord, une petite
illustration du problème avec la classe
`PiecewiseRegression <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression.html>`_
implémentée selon l'API de :epkg:`scikit-learn`.

.. toctree::
    :maxdepth: 1

    ../notebooks/piecewise_linear_regression

.. image:: piecewise/piecenaive.png
    :width: 250

Cette régression par morceaux est obtenue grâce à un arbre
de décision. Celui-ci trie le nuage de points :math:`(X_i, Y_i)`
par ordre croissant selon les *X*, soit :math:`X_i \leqslant X_{i+1}`.
L'arbre coupe en deux lorsque la différence des erreurs quadratiques est
maximale, erreur quadratiques obtenus en approximation *Y* par sa moyenne
sur l'intervalle considéré. On note l'erreur quadratique :

.. math::

    \begin{array}{rcl}
    C_(i,j) &=& \frac{1}{j - i + 1} \sum_{i \leqslant k \leqslant j} Y_i \\
    D_(i,j) &=& \frac{1}{j - i + 1} \sum_{i \leqslant k \leqslant j} Y^2_i \\
    E_(i,j) &=& \frac{1}{j - i + 1} \sum_{i \leqslant k \leqslant j} ( Y_i - C(i,j))^2 =
    \frac{1}{j - i + 1} \sum_{i \leqslant k \leqslant j} Y_i^2 - C(i,j)^2 = D(i,j) - C(i,j)^2
    \end{array}

La dernière ligne applique la formule :math:`\var{X} = \esp{X^2} - \esp{X}^2`
qui est facile à redémontrer.
L'algorithme de l'arbre de décision coupe un intervalle en
deux en détermine l'indice *k* qui minimise la différence :

.. math::

    \Delta_k = E(1, n) - (E(1, k) + E(k+1, n))

L'arbre de décision optimise la construction d'une fonction
en escalier qui représente au mieux le nuage de points,
les traits verts sur le graphe suivant, alors qu'il faudrait
choisit une erreur quadratique qui correspondent aux traits
oranges.

.. image:: piecewise/piecenaive2.png
    :width: 250

Il suffirait donc de remplacer l'erreur *E* par celle obtenue
par une régression linéaire. Mais si c'était aussi simple,
l'implémentation de :epkg:`sklearn:tree:DecisionTreeRegressor`
la proposerait. Alors pourquoi ?
La raison principale est que cela coûte trop cher en
temps de calcul. Pour trouver l'indice *k*, il faut calculer
toutes les erreurs :math:`E(1,k)` :math:`E(k+1,n)`, ce qui
coûte très cher lorsque cette erreur est celle d'une régression
linéaire parce qu'il est difficile de simplifier la différence :

.. math::

    \begin{array}{rcl}
    \Delta_{k} - \Delta_{k-1} &=&  - (E(1, k) + E(k+1, n)) + (E(1, k-1) + E(k, n)) \\
    &=&  E(1, k-1) - E(1, k) + E(k, n) - E(k+1, n)
    \end{array}

On s'intéresse au terme :math:`E(1, k-1) - E(1, k)` :

.. math::

    \begin{array}{rcl}
    C_(1,k-1) - C_(1,k) &=& \frac{1}{k-1} \sum_{1 \leqslant i \leqslant k-1} Y_i
    - \frac{1}{k} \sum_{1 \leqslant i \leqslant k} Y_i \\
    &=& (\frac{1}{k-1} - \frac{1}{k}) \sum_{1 \leqslant i \leqslant k-1} Y_i - \frac{Y_k}{k} \\
    &=& \frac{1}{k(k-1)} \sum_{1 \leqslant i \leqslant k-1} Y_i- \frac{Y_k}{k} \\
    &=& \frac{1}{k} C(1,k-1) - \frac{Y_k}{k}
    \end{array}

On en déduit que :

.. math::

    \begin{array}{rcl}
    E(1, k-1) - E(1, k) &=& \frac{1}{k} D(1,k-1) - \frac{Y_k^2}{k} +
    (C_(1,k-1) - C_(1,k))(C_(1,k-1) + C_(1,k)) \\
    &=& \frac{1}{k} D(1,k-1) - \frac{Y_k^2}{k} + \pa{\frac{1}{k} C(1,k-1) - \frac{Y_k}{k}}
    \pa{\frac{Y_k}{k} - \frac{1}{k} C(1,k-1) + 2 C(1,k-1)}
    \end{array}

On voit que cette formule ne fait intervenir que :math:`C(1,k-1), D(1,k-1), Y_k`,
elle est donc très rapide à calculer et c'est pour cela qu'apprendre un arbre
de décision peut s'apprendre en un temps raisonnable. Cela repose sur la possibilité
de calculer le critère optimisé par récurrence. On voit également que ces formules
ne font pas intervenir *X*, elles sont donc généralisables au cas
multidimensionnel. Il suffira de trier les couples :math:`(X_i, Y_i)`
selon chaque dimension et déterminer le meilleur seuil de coupure
d'abord sur chacune des dimensions puis de prendre le meilleur
de ces seuils sur toutes les dimensions. Le problème est résolu.

Le notebook :epkg:`Custom Criterion for DecisionTreeRegressor`
implémente une version pas efficace du critère
`MSE <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_
et compare la vitesse d'exécution avec l'implémentation de :epkg:`scikit-learn`.
Il implémente ensuite le calcul rapide de *scikit-learn* pour
montrer qu'on obtient un temps comparable.
Le résultat est sans équivoque. La version rapide n'implémente pas
:math:`\Delta_{k} - \Delta_{k-1}` mais plutôt les sommes
:math:`\sum_1^k w_i Y_i`, :math:`\sum_1^k w_i Y_i^2` dans un sens
et dans l'autre. En gros,
le code stocke les séries des numérateurs et des dénominateurs
pour les diviser au dernier moment.

Le cas d'une régression est plus complexe. Prenons d'abord le cas
où il n'y a qu'un seule dimension,
il faut d'abord optimiser le problème :

.. math::

    E(1, n) = \min_{a,b} = \sum_{k=1}^n (a X_k + b - Y_k)^2

On dérive pour aboutir au système d'équations suivant :

.. math::

    \begin{array}{rcl}
    \frac{\partial E(1,n)}{\partial a} &=& 0 = \sum_{k=1}^n X_k(a X_k + b - Y_k) \\
    \frac{\partial E(1,n)}{\partial b} &=& 0 = \sum_{k=1}^n a X_k + b - Y_k
    \end{array}

Ce qui aboutit à :

.. math::

    \begin{array}{rcl}
    a(1, n) &=& \frac{\sum_{k=1}^n X_kY_k - \pa{\sum_{k=1}^n X_k}\pa{\sum_{k=1}^n Y_k} }
    {\sum_{k=1}^n X_k^2 -\pa{\sum_{k=1}^n X_k}^2 } \\
    b(1, n) &=& \sum_{k=1}^n Y_k - a \pa{\sum_{k=1}^n X_k}
    \end{array}

Pour construire un algorithme rapide pour apprendre un arbre de décision
avec cette fonction de coût, il faut pouvoir calculer
:math:`a(1, k)` en fonction de :math:`a(1, k-1), b(1, k-1), X_k, Y_k`
ou d'autres quantités intermédiaires qui ne font pas intervenir
les valeurs :math:`X_{i<k} < Y_{i<k}`. D'après ce qui précède,
cela paraît tout-à-fait possible. Mais dans le
`cas multidimensionnel
<https://fr.wikipedia.org/wiki/R%C3%A9gression_lin%C3%A9aire#Estimateur_des_moindres_carr%C3%A9s>`_,
il faut déterminer la vecteur *A* qui minimise :math:`\sum_{k=1}^n \norme{Y - XA}^2`
ce qui donne :math:`A = (X'X)^{-1} X' Y`. Si on note :math:`M_{1..k}` la matrice
*M* tronquée pour ne garder que ses *k* premières lignes, il faudrait pouvoir
calculer rapidement :

.. math::

    A_{k-1} - A_k = (X_{1..k-1}'X_{1..k-1})^{-1} X'_{1..k-1} Y_{1..k-1} -
    (X_{1..k}'X_{1..k})^{-1} X'_{1..k} Y_{1..k}

Pas simple... La documentation de :epkg:`sklearn:tree:DecisionTreeRegressor`
ne mentionne que deux critères pour apprendre un arbre de décision
de régression, *MSE* pour
:epkg:`sklearn:metrics:mean_squared_error` et *MAE* pour
:epkg:`sklearn:metrics:mean_absolute_error`. Les autres critères n'ont
probablement pas été envisagés car il n'existe pas de façon efficace
de les implémenter. L'article [Acharya2016]_ étudie la possibilité
de ne pas calculer la matrice :math:`A_k` pour tous les *k*.
Mais ce n'est pas la direction choisie pour cet exposé.

Implémentation naïve d'une régression linéaire par morceaux
===========================================================

On part du cas général qui écrit la solution d'une régression
linéaire comme étant la matrice :math:`A = (X'X)^{-1} X' Y`
et on adapte l'implémentation de :epkg:`scikit-learn` pour
optimiser l'erreur quadratique obtenue. Ce n'est pas simple mais
pas impossible. Il faut entrer dans du code :epkg:`cython` et, pour
éviter de réécrire une fonction qui multiplie et inverse uen matrice,
on peut utiliser la librairie :epkg:`LAPACK`. Je ne vais pas plus loin
ici car cela serait un peu hors sujet mais ce n'était pas une partie
de plaisir. Cela donne :
`piecewise_tree_regression_criterion_linear.pyx
<https://github.com/sdpython/mlinsights/blob/master/src/mlinsights/mlmodel/piecewise_tree_regression_criterion_linear.pyx>`_
C'est illustré toujours par le notebook
:epkg:`DecisionTreeRegressor optimized for Linear Regression`.

Aparté sur la continuité de la régression linéaire par morceaux
===============================================================

.. index:: optimisation sous contrainte, continuité

Approcher la fonction :math:`y=f(x) + \epsilon` quand *x* et *y*
sont réels est un problème facile, trop facile... A voir le dessin,
précédent, il est naturel de vouloir recoller les morceaux lorsqu'on
passe d'un segment à l'autre. Il s'agit d'une optimisation sous contrainte.
Il est possible également d'ajouter une contrainte de régularisation
qui tient compte de cela. On exprime cela comme suit avec une régression
linéaire à deux morceaux.

.. math::

    E = \sum_{X_i \leqslant t} (a_1 X_i + b_1 - y)^2 +
    \sum_{X_i \geqslant t} (a_2 X_i + b_2 - y)^2 +
    \lambda (a_1 t + b_1 - a_2 t - b)^2

Le cas multidimensionnel est loin d'être aussi simple. Avec une
dimension, chaque zone a deux voisines. En deux dimensions,
chaque zone peut en avoir plus de deux. La figure suivante
montre une division de l'espace dans laquelle la zone centrale
a cinq voisins.

.. image:: piecewise/voisin.png
    :width: 200

Peut-on facilement approcher une fonction :math:`z = f(x,y) + \epsilon`
par un plan en trois dimensions ? A moins que tous les sommets soient
déjà dans le même plan, c'est impossible. La zone en question n'est
peut-être même pas convexe. Une régression linéaire par morceaux
et continue en plusieurs dimensions n'est pas un problème facile.
Cela n'empêche pas pour autant d'influencer la détermination de chaque
morceaux avec une contrainte du type de celle évoquée plus haut
mais pour écrire la contrainte lorsque les zones sont construites
à partir des feuilles d'un arbre de décision, il faut déterminer
quelles sont les feuilles voisines.
Et ça c'est un problème intéressant !

Régression linéaire et corrélation
==================================

On reprend le calcul multidimensionnel mais on s'intéresse au
cas où la matrice :math:`X'X` est diagonale qui correspond au cas
où les variables :math:`X_1, ..., X_C` ne sont pas corrélées.
Si :math:`X'X = diag(\lambda_1, ..., \lambda_C) = diag(\sum_{k=1}^n X^2_{k1}, ..., \sum_{k=1}^n X^2_{kC})`,
la matrice :math:`A` s'exprime plus simplement :math:`A = D^{-1} X' Y`.
On en déduit que :

.. math::

    a_c = \frac{\sum_{k=1}^n X_{kc} Y_k}{\sum_{k=1}^n X^2_{kc}} =
    \frac{\sum_{k=1}^n X_{kc} Y_k}{\lambda_c}

Cette expression donne un indice sur la résolution d'une régression linéaire
pour laquelle les variables sont corrélées. Il suffit d'appliquer d'abord une
`ACP <https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales>`_
(Analyse en Composantes Principales) et de calculer les coefficients
:math:`a_c` associés à des valeurs propres non nulles. On écrit alors
:math:`X'X = P'DP` où la matrice *P* vérifie :math:`P'P = I`.

Idée de l'algorithme
====================

On s'intéresser d'abord à la recherche d'un meilleur point de coupure.
Pour ce faire, les éléments :math:`(X_i, y_i)` sont triés le plus souvent
selon l'ordre défini par une dimension. On note *E* l'erreur de prédiction
sur cette échantillon :math:`E = \min_\beta \sum_k (X_k \beta - y_k)^2`.
On définit ensuite :math:`E(i, j) = \min_\beta \sum_{k=i}^j (X_k \beta - y_k)^2`.
D'après cette notation, :math:`E = E(1,n)`. La construction de l'arbre
de décision passe par la détermination de :math:`k^*` qui vérifie :

.. math::

    \begin{array}{rcl}
    E(1,k^*) + E(k^*+1, n) &=& \min_k E(1,k) + E(k+1, n) \\
    &=& \min_k \pa{ \min_{\beta_1} \sum_{l=1}^k (X_l \beta_1 - y_l)^2 +
    \min_{\beta_2} \sum_{l=k+1}^n (X_l \beta_2 - y_l)^2}
    \end{array}

Autrement dit, on cherche le point de coupure qui maximise la différence
entre la prédiction obtenue avec deux régressions linéaires plutôt qu'une.
On sait qu'il existe une matrice *P* qui vérifie :

.. math::

    PP' = 1 \text{ et } (XP)'(XP) = P'X'XP = D = Z'Z

Où :math:`D=diag(d_1, ..., d_C)` est une matrice
diagonale. On a posé :math:`Z = XP`,
donc :math:`d_a = <Z_a, Z_a>`.
On peut réécrire le problème
de régression comme ceci :

.. math::

    \beta^* = \arg\min_\beta \sum_i \norm{ y_i - X_i\beta} =
    \arg\min_\beta \norm{Y - X\beta}

Comme :math:`X = ZP'` :

.. math::

    \norm{Y - X\beta} = \norm{Y - X\beta} = \norm{Y - ZP'\beta} =
    \norm{Y - Z\gamma}

Avec :math:`\gamma = P'\beta`. C'est la même régression
après un changement de repère et on la résoud de la même manière :

.. math::

    \gamma^* = (Z'Z)^{-1}Z'Y = D^{-1}Z'Y

La notation :math:`M_i` désigne la ligne *i* et
:math:`M_{[k]}` désigne la colonne.
On en déduit que le coefficient de la régression
:math:`\gamma_k` est égal à :

.. math::

    \gamma_k = \frac{<Z_{[k]},Y>}{<Z_{[k]},Z_{[k]}>} =
    \frac{<(XP')_{[k]},Y>}{<(XP')_{[k]},(XP')_{[k]}>}

On en déduit que :

.. math::

    \norm{Y - X\beta} = \norm{Y - \sum_{k=1}^{C}Z_{[k]}\frac{<Z_{[k]},Y>}{<Z_{[k]},Z_{[k]}>}} =
    \norm{Y - \sum_{k=1}^{C}(XP')_{[k]}\frac{<(XP')_{[k]},Y>}{<(XP')_{[k]},(XP')_{[k]}>}}

.. mathdef::
    :title: Arbre de décision optimisé pour les régressions linéaires
    :tag: Algorithme
    :lid: algo_decision_tree_mselin

    On dipose qu'un nuage de points :math:`(X_i, y_i)` avec
    :math:`X_i \in \R^d` et :math:`y_i \in \R`. Les points sont
    triés selon une dimension. On note *X* la matrice composées
    des lignes :math:`X_1, ..., X_n` et le vecteur colonne
    :math:`(y_1, ..., y_n)`.
    Il existe une matrice :math:`P` telle que :math:`P'P = I`
    et :math:`X'X = P'DP` avec *D* une matrice diagonale.
    On note :math:`X_{a..b}` la matrice constituée des lignes
    *a* à *b*. On calcule :

    .. math::

        MSE(X, y, a, b) = \norm{Y - \sum_{k=1}^{C}(X_{a..b}P')_{[k]}
        \frac{<(X_{a..b}P')_{[k]},Y>}{<(X_{a..b}P')_{[k]},(X_{a..b}P')_{[k]}>}}^2

    Un noeud de l'arbre est construit en choisissant le point
    de coupure qui minimise :

    .. math::

        MSE(X, y, 1, t) + MSE(X, y, t+1, n)

Un peu plus en détail dans l'algorithme
=======================================

J'ai pensé à plein de choses pour aller plus loin car l'idée
est de quantifier à peu près combien on pert en précision en utilisant
des vecteurs propres estimés avec l'ensemble des données sur une partie
seulement. Je me suis demandé si les vecteurs propres d'une matrice
pouvait être construit à partir d'une fonction continue de la matrice
symétrique de départ. A peu près vrai mais je ne voyais pas une façon
de majorer cette continuité. Ensuite, je me suis dit que les vecteurs
propres de :math:`X'X` ne devaient pas être loin de ceux de :math:`X_\sigma'X_\sigma`
où :math:`\sigma` est un sous-échantillon aléatoire de l'ensemble
de départ. Donc comme il faut juste avoir une base de vecteurs
orthogonaux, je suis passé à l'`orthonormalisation de Gram-Schmidt
<https://fr.wikipedia.org/wiki/Algorithme_de_Gram-Schmidt>`_.
Il n'a pas non plus ce défaut de permuter les dimensions ce qui rend
l'observation de la continuité a little bit more complicated comme
le max dans l'`algorithme de Jacobi
<https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm>`_.
L'idée est se servir cette orthonormalisation pour construire
la matrice *P* de l'algortihme.

La matrice :math:`P \in \mathcal{M}_{CC}` est constituée de
*C* vecteurs propres :math:`(P_{[1]}, ..., P_{[C]})`. Avec les notations que
j'ai utilisées jusqu'à présent :
:math:`X_{[k]} = (X_{1k}, ..., X_{nk})`.
On note la matrice identité :math:`I_C=I`.

.. math::

    \begin{array}{rcl}
    U_{[1]} &=& \frac{ X_{[1]} }{ \norme{X_{[1]}} } \\
    P_{[1]} &=& \frac{ I_{[1]} }{ \norme{X_{[1]}} } \\
    U_{[2]} &=& \frac{ X_{[2]} - <X_{[2]}, U_{[1]}> U_{[1]} }
    { \norme{X_{[2]} - <X_{[2]}, U_{[1]}> U_{[1]}} } \\
    P_{[2]} &=& \frac{ I_{[2]} - <X_{[2]}, U_{[1]}> U_{[1]} }
    { \norme{X_{[2]} - <X_{[2]}, U_{[1]}> U_{[1]}} } \\
    ... && \\
    U_{[k]} &=& \frac{ X_{[k]} - \sum_{i=1}^{k-1} <X_{[k]}, U_{[i]}> U_{[i]} }
    { \norme{ X_{[2]} - \sum_{i=1}^{k-1} <X_{[k]}, U_{[i]}> U_{[i]} } } \\
    P_{[k]} &=& \frac{ I_{[k]} - \sum_{i=1}^{k-1} <X_{[k]}, U_{[i]}> U_{[i]} }
    { \norme{ X_{[2]} - \sum_{i=1}^{k-1} <X_{[k]}, U_{[i]}> U_{[i]} } } \\
    \end{array}

La matrice *U* vérifie :math:`U'U` puisque les vecteurs sont
construits de façon à être orthonormés. Et on vérifie que
:math:`XP = U` et donc :math:`PXX'P' = I`.
C'est implémenté par la fonction
:func:`gram_schmidt <mlstatpy.ml.matrices.gram_schmidt>`.

.. runpython::
    :showcode:

    import numpy
    from mlstatpy.ml.matrices import gram_schmidt

    X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
    U, P = gram_schmidt(X.T, change=True)
    U, P = U.T, P.T
    m = X @ P
    D = m.T @ m
    print(D)

Cela débouche sur une autre formulation du calcul
d'une régression linéaire à partir d'une orthornormalisation
de Gram-Schmidt qui est implémentée dans la fonction
:func:`linear_regression <mlstatpy.ml.matrices.linear_regression>`.

.. runpython::
    :showcode:

    import numpy
    from mlstatpy.ml.matrices import linear_regression

    X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
    y = numpy.array([1, 1.3, 3.9])
    beta = linear_regression(X, y, algo="gram")
    print(beta)

L'avantage est que cette formulation s'exprime
uniquement à partir de produits scalaires.
Voir le notebook :ref:`regressionnoinversionrst`.

Implémentation
==============

Bilbiographie
=============

.. [Acharya2016] `Fast Algorithms for Segmented Regression <https://arxiv.org/abs/1607.03990>`_,
    Jayadev Acharya, Ilias Diakonikolas, Jerry Li, Ludwig Schmidt, :epkg:`ICML 2016`
