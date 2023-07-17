
.. _l-quantization:

============
Quantization
============

.. contents::
    :local:

Un problème simple
==================

Les réseaux de neurones (deep learning) sont de plus en plus gros
et nécessitent de plus en plus de puissance de calcul. La
**quantization** est une façon de contourner en réduisant
la mémoire nécessaire pour stocker les coefficients et le
temps de calcul avec les dernières cartes graphiques.
La quantization est équivalent à une discrétisation.
Voir aussi :
`A Survey of Quantization Methods for Efficient
Neural Network Inference <https://arxiv.org/pdf/2103.13630.pdf>`_.

Les produits matriciels sont fréquents dans les réseaux de neurones où
on multiple les entrées *X* d'une couche de neurones avec les coefficients
*A* : :math:`X B`. Lors de l'apprentissage, la matrice B est apprise soit en float 32 bit
soit en float 16 bit puis elle est discrétisée sur 8 bit, soit 256 valeurs.
Si on note *q(B)* la forme *discrétisée* de *B*, le plus simple de minimiser
:math:`\norm{B - q(B)}^2`. On pourrait également se pencher sur la minimisation
de :math:`X (B - q(B))^2` mais la matrice *X* vient des données.
Il faudrait soit prendre des hypothèses de distribution sur X
ou optimiser sur un jeu de données :math:`(X_i)_i`.

On considère le cas simple qui consiste à minimiser
:math:`\norm{B - q(B)}^2`.

Discrétisation sur un octet
===========================

On discrétise sur 256 valeurs espacées de façon régulière sur un intervalle.
Mais les coefficients ne sont pas tous situés dans un même intervalle.
On doit alors trouver les meilleurs paramètres :math:`z` et :math:`\lambda`
qui définissent la quantization au travers de deux fonctions. On note
:math:`c_{0}^{255}(x)=\max(0, \min(255, x))` la fonction qui *x*
dans l'intervalle *[0, 255]*, 0 à gauche, 255 à droite.

.. math::

    \begin{array}{rcl}
    q_1(z, \lambda, x) &=& c_{0}^{255}\pa{\intf_{i8}{\frac{x}{\lambda}} + z} \text{ quantization}\\
    q_2(z, \lambda, i) &=& \lambda(i - z) \text{ déquantization} \\
    q(z, \lambda, x) &=& q_2(z, \lambda, q_1(z, \lambda, x)) \\
    &=& \lambda\pa{c_{0}^{255}\pa{\intf_{i8}{\frac{x}{\lambda}} + z} - z} \\
    &=& \lambda\intf_{i8,z}{\frac{x}{\lambda}}
    \end{array}

La fonction :math:`\intf_{i8,z}{x}` est la partie entière asociée à la fonction
:math:`c_{0}^{255}(i)`.

.. math::

    \norm{B - q(z,\lambda,B)}^2 = \sum_{ij} \pa{b_{ij} - \lambda\intf_{i8,z}{\frac{x}{\lambda}}}^2

Le problème est la fonction :math:`\intf_{i8,z}{.}` qui n'est pas dérivable.
C'est un problème d'optimisation discrète. Le paramètre :math:`\lambda`
est appelé *scale* ou *échelle*. Il peut y en avoir un ou plusieurs
mais dans ce cas, on considère les différentes parties de *B*
qui sont quantizées avec les mêmes paramètres :math:`\lambda` et *z*

Cette quantization est appelée *quantization linéaire*. Elle est privilégiée
car elle est très rapide et la transformation inverse (ou déquantization)
l'est tout autant.

Discrétisation sur une float 8
==============================

Un float 8 est un réel codé sur 8 bits. Il y a plusieurs variantes.
Nous allons considérer la version *E4M3FN*, :math:`S.1111.111_2` :

* 1 bit pour le signe
* 4 bits pour l'exposant
* 3 bits pour la mantisse

Et la valeur réelle est :

* si l'exposant est nul, 
  :math:`(-1)^S 2^{\sum_{i=3}^6 b_i 2^{i-3}- 7}\left(1+\sum_{i=0}^2 b_i 2^{i-3}\right)`
* sinon :math:`(-1)^S 2^{-6} \sum_{i=0}^2 b_i 2^{i-3}`
* le réel vaut NaN s'il suit le format : :math:`S.1111.111_2` (255 ou 127),
* le réel vaut zéro s'il suit le format : :math:`S.0000.000_2` (0 ou 128)

Les valeurs ne sont plus uniformément espacées mais il y en a toujours entre 252 et 255
selon les formats de float 8 et on cherche toujours à trouver les meilleurs
paramètres :math:`\lambda` et *z*. La formule est quasiment la même. On n'arrondit
plus à l'entier inférieur (ou le plus proche) mais au float 8
inférieur (ou le plus proche).

.. math::

    \norm{B - q(z,\lambda,B)}^2 = \sum_{ij} \pa{b_{ij} - \lambda\intf_{f8,z}{\frac{x}{\lambda}} }^2

Optimisation
============

L'idée est de traiter la discrétisation sur un ensemble fini de valeurs,
quel qu'il soit, des entiers ou des réels codés sur 8 bits. On note Cette
ensemble :math:`(d_1, ..., d_n)`. On réécrit le problème d'optimisation :

.. math::

    \begin{array}{rcl}
    \norm{B - q(z,\lambda,B)}^2 &=& \sum_{ij} \pa{b_{ij} - \lambda\intf_{f8,z}{\frac{x}{\lambda}} }^2 \\
    &=& \sum_{k=1}^{n} \sum_{ij} \pa{b_{ij} - \lambda\intf_{f8}{\frac{x}{\lambda}} }^2
    \indicatrice{\intf_{f8}{\frac{x}{\lambda}} = d_k}
    \end{array}



