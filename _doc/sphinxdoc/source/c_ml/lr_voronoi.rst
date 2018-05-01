
====================================================
Régression logistique, diagramme de Voronoï, k-Means
====================================================

.. index:: régression logistique, diagramme de Voronoï, Voronoï

Ce qui suit explore les liens entre une régression logistique,
les diagrammes de Voronoï pour construire un classifieur
qui allient la régression logistique et les clustering type k-means.
Le point de départ est une conjecture : les régions
créées par une régression logisitique sont convexes.

.. contents::
    :local:

Diagramme de Voronoï
====================

.. index:: diagramme de Voronoï

Un `diagramme de Voronoï <https://fr.wikipedia.org/wiki/Diagramme_de_Vorono%C3%AF>`_
est le diagramme issu des intersections des médiatrices entre :math:`n` points.

.. image:: lrvor/Coloured_Voronoi_2D.png
    :width: 200

On définit un ensemble de points :math:`(X_1, ..., X_n)`.
La zone d'influence de chaque point est défini par
:math:`V(X_i) = \{ x | d(x, X_i) \leqslant d(x, X_j) \forall j\}`.
Si *d* est la distance euclidienne, la frontière entre deux
points :math:`X_i, X_j` est un segment sur la droite d'équation
:math:`d(x, X_i) = d(x, X_j)` :

.. math::

    \begin{array}{ll}
    &\norme{x-X_i}^2 - \norme{x-X_j}^2 = 0 \\
    \Longrightarrow & \norme{x}^2 - 2 \scal{x}{X_i} + \norme{X_i}^2 - (\norme{x}^2 - 2 \scal{x}{X_j} + \norme{X_j}^2) = 0 \\
    \Longrightarrow & 2 \scal{x}{X_j - X_i} + \norme{X_i}^2 - \norme{X_j}^2 = 0 \\
    \Longrightarrow & 2 \scal{x}{X_j - X_i} + \frac{1}{2} \scal{X_i + X_j}{X_i - X_j} = 0 \\
    \Longrightarrow & \scal{x - \frac{X_i + X_j}{2}}{X_i - X_j} = 0
    \end{array}

Ce système constitue :math:`\frac{n(n-1)}{2}` droites ou hyperplans si
l'espace vectoriel est en dimension plus que deux.
Le diagramme de Voronoï est formé par des segments de chacune
de ces droites. On peut retourner le problème. On suppose
qu'il existe :math:`\frac{n(n-1)}{2}` hyperplans,
existe-t-il *n* points de telle sorte que les hyperplans
initiaux sont les frontières du diagramme de Voronoï formé
par ces *n* points ?

**à compléter**

Régression logistique
=====================

:epkg:`scikit-learn` a rendu populaire le jeu de données
`Iris <http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py>`_
qui consiste à classer des fleurs en trois classes
en fonction des dimensions de leurs pétales.

.. image:: lrvor/iris.png

.. runpython::
    :showcode:
    :warningout: ImportWarning

    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data[:, :2], data.target

    from sklearn.linear_model import LogisticRegression
    clr = LogisticRegression()
    clr.fit(X, y)

    print(clr.coef_)
    print(clr.intercept_)

La fonction de prédiction est assez simple :
:math:`f(x) = Ax + B`. La classe d'appartenance
du point *x* est déterminé par :math:`\max_i f(x)_i`.
La frontière entre deux classes *i, j* est définie
par les deux conditions :
:math:`\max_k f(x)_k = f(x)_i = f(x)_j`.
On retrouve bien :math:`\frac{n(n-1)}{2}` hyperplans.
On définit la matrice *A* comme une matrice
ligne :math:`(L_1, ..., L_n)` où *n* est le nombre
de classes. L'équation de l'hyperplan entre deux classes devient :

.. math::

    \begin{array}{ll}
    & L_i X + B_i = L_j X + B_j \\
    \Longleftrightarrow & (L_i - L_j) X + B_i - B_j = 0 \\
    \Longleftrightarrow & \scal{L_i - L_j}{X} + B_i - B_j = 0 \\
    \Longleftrightarrow & \scal{L_i - L_j}{X - \frac{L_i + L_j}{2}} + \scal{L_i - L_j}{\frac{L_i + L_j}{2}} + B_i - B_j = 0 \\
    \Longleftrightarrow & \scal{L_i - L_j}{X - \frac{L_i + L_j}{2}} + \frac{1}{2}\norme{L_i}^2 - \frac{1}{2}\norme{L_j}^2 + B_i - B_j = 0 \\
    \Longleftrightarrow & \scal{L_i - L_j}{X - \frac{L_i + L_j}{2}} + \frac{1}{2}\norme{L_i}^2 + B_i - (\frac{1}{2}\norme{L_j}^2 + B_j) = 0
    \end{array}

Il y a peu de chance que cela fonctionne comme cela.
Avant de continuer, assurons-nous que les régions associées
aux classes sont convexes. C'est une condition nécessaire mais
pas suffisante pour avoir un diagramme de Voronoï.

Soit :math:`X_1` et :math:`X_2` appartenant à la classe *i*.
On sait que que :math:`\forall k, L_i X_1 + B_i \geqslant L_k X_1 + B_k`
et :math:`\forall k, L_i X_2 + B_i \geqslant L_k X_2 + B_k`.
On considère un point :math:`X` sur le segment :math:`[X_1, X_2]`, donc il existe
:math:`\alpha, \beta \geqslant 0` tel que :math:`X = \alpha X_1 + \beta X_2` et
:math:`\alpha + \beta = 1`. On vérifie que :

.. math::

    \begin{array}{ll}
    & L_i X + B_i = L_i (\alpha X_1 + \beta X_2) + B_i = \alpha(L_i X_1 + B_i) + \beta(L_i X_2 + B_i) \\
    \geqslant & \alpha(L_k X_1 + B_k) + \beta(L_k X_2 + B_k) = L_k (\alpha X_1 + \beta X_2) + B_k \forall k
    \end{array}

Donc le point *X* appartient bien à classe *i* et celle-ci est convexe.
La régression logistique forme une partition convexe de l'espace
des features.

.. mathdef::
    :title: convexité des classes formées par une régression logistique
    :tag: Théorème

    On définit l'application :math:`\mathbb{R}^d \rightarrow \mathbb{N}`
    qui associe la plus grande coordonnées
    :math:`f(X) = \arg \max_k (AX + B)_k`.
    *A* est une matrice :math:`\mathcal{M}_{dc}`,
    *B* est un vecteur de :math:`\mathbb{R}^d`,
    *c* est le nombre de parties.
    L'application *f* définit une partition convexe
    de l'espace vectoriel :math:`\mathbb{R}^d`.

Revenons au cas de Voronoï. La classe prédite dépend de
:math:`\max_k (Ax + B)_k`. On veut trouver *n* points
:math:`(P_1, ..., P_n)` tels que chaque couple :math:`(P_i, P_j)`
soit équidistant de la frontière qui sépare leurs classes.
Il faut également les projections des deux points sur
la frontière se confondent et donc que les vecteurs
:math:`P_i - P_j` et :math:`L_i - L_j` sont colinéaires.

.. math::

    \begin{array}{ll}
    &\left\{\begin{array}{l}\scal{L_i - L_j}{P_i} + B_i - B_j = - \pa{ \scal{L_i - L_j}{P_j} + B_i - B_j } \\
    P_i-  P_j - \scal{P_i - P_j}{\frac{L_i-L_j}{\norm{L_i-L_j}}} \frac{L_i-L_j}{\norm{L_i-L_j}}=0 \end{array} \right.
    \\
    \Longleftrightarrow & \left\{\begin{array}{l}\scal{L_i - L_j}{P_i + P_j} + 2 (B_i - B_j) = 0 \\
    P_i-  P_j - \scal{P_i - P_j}{\frac{L_i-L_j}{\norm{L_i-L_j}}} \frac{L_i-L_j}{\norm{L_i-L_j}}=0 \end{array} \right.
    \end{array}

La seconde équation en cache en fait plusieurs puisqu'elle est valable
sur plusieurs dimensions mais elles sont redondantes.
Il suffit de choisir un vecteur :math:`u_{ij}` non perpendiculaire
à :math:`L_i - L_j` de sorte que
qui n'est pas perpendiculaire au vecteur :math:`L_i - L_j` et de
considérer la projection de cette équation sur ce vecteur.
C'est pourquoi on réduit le système au suivant qui est
équivalent au précédent si le vecteur :math:`u_{ij}` est bien choisi.

.. math::

    \begin{array}{ll}
    \Longrightarrow & \left\{\begin{array}{l}\scal{L_i - L_j}{P_i + P_j} + 2 (B_i - B_j) = 0 \\
    \scal{P_i-  P_j}{u_{ij}} - \scal{P_i - P_j}{\frac{L_i-L_j}{\norm{L_i-L_j}}} \scal{\frac{L_i-L_j}{\norm{L_i-L_j}}}{u_{ij}}=0
    \end{array} \right.
    \end{array}

Faisons un peu de géométrie avant de résoudre ce problème car celui-ci
a dans la plupart des cas plus d'équations que d'inconnues.
Chaque frontière entre deux classes est la médiatrice d'un segment
:math:`[P_i, P_j]`. Le dessin suivant trace un diagramme de Voronoï à
trois points. L'intersection est le centre des médiatrices du triangle
formé par les points de Voronoï. Pour les trouver, on trace un cercle,
n'importe lequel, puis une droite perpendiculaire à l'une des médiatrice.
On obtient deux points. Le troisième est obtenu en traçant une seconde
perpendiculaire et par construsction, la troisième droite est perpendiculaire
à la troisième médiatrice. Et on nomme les angles.

.. image:: lrvor/vor2.png
    :width: 200

.. image:: lrvor/vor4.png
    :width: 300

Les triangles formés par les côtés jaunes sont isocèles. On en déduit que
:math:`a + b + c = 2\pi = 2(x + y + z)`. On en déduit aussi que :

.. math::

    \begin{array}{rcl}
    x + y &=& a \\
    y + z &=& c \\
    x + z &=& b
    \end{array}

On en conclut que :math:`a + b + c = 2\pi = 2(x + y + z) = 2(x + c)` et
:math:`x = \pi - c`. Il existe une infinité de triplets de 3 points
qui aboutissent à ce diagramme de Voronoï. Il suffit de changer
la taille du cercle. On montre aussi qu'en dimension 2 et 3 classes,
il existe toujours une solution au problème posé.
Maintenant, si on considère la configuration suivante avec des points
disposés de telle sorte que le diagramme de Voronoï est un maillage
hexagonal. :math:`a=b=c=\frac{2\pi}{3}` et :math:`x=y=z=\frac{\pi}{3}`.
Il n'existe qu'un ensemble de points qui peut produire ce maillage
comme diagramme de Voronoï. Mais si on ajoute une petite zone
(dans le cercle vert ci-dessous), il est impossible que ce diagramme
soit un diagramme de Voronoï bien que cela soit une partition convexe.

.. image:: lrvor/hexa.png
    :width: 200

.. image:: lrvor/hexa2.png
    :width: 200

On revient à la détermination du diagramme de Voronoï associé à
une régression logistique. On a montré qu'il n'existe pas tout le temps,
qu'il peut y avoir une infinité de solutions et qu'il est la solution
d'un système d'équations linéaires.

Notebooks
=========

.. index:: boule unité

Le notebook qui suit reprend les différents
éléments théoriques présentés ci-dessus. Il
continue l'étude d'une régression logistique
et donne une intuition de ce qui marche ou pas
avec un tel modèle. Notamment, le modèle est plus
performant si les classes sont situées sur la boule
unité de l'espace des features.

.. toctree::
    :maxdepth: 1

    ../notebooks/logreg_voronoi
