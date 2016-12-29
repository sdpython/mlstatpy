
.. _nn-classification:

La classification
=================

Comme la régression, la classification consiste aussi à trouver le
lien entre une variable :math:`X` et une variable aléatoire discrète
suivant une `loi multinomiale <https://fr.wikipedia.org/wiki/Loi_multinomiale>`_ :math:`Y`.

.. mathdef::
    :title: Classification
    :tag: Problème
    :lid: probleme_classification

    Soit une variable aléatoire :math:`X`
    et une variable aléatoire discrète :math:`Y`,
    l'objectif est d'approximer la fonction :math:`\esp\pa{Y | X} = f\pa{X}`.
    Les données du problème sont
    un échantillon de points : :math:`\acc { \pa{ X_{i},Y_{i} } | 1 \infegal i \infegal N }`
    avec :math:`\forall i \in \ensemble{1}{N}, \; Y_i \in \ensemble{1}{C}`
    et un modèle paramétré avec :math:`\theta` :

    .. math::

             \forall  i \in \intervalle{1}{N}, \;  \forall  c \in \intervalle{1}{C}, \;
                        \pr { Y_i = c | X_i, \theta} = h \pa{\theta,X_i,c }

    avec :math:`n \in \N`,  :math:`h` est une fonction de paramètre
    :math:`\theta` à valeur dans :math:`\cro{0,1}` et vérifiant la
    contrainte : :math:`\sum_{c=1}^C h(\theta,X,c) = 1`.
		

Le premier exemple
est une classification en deux classes, elle consiste à découvrir le lien qui
unit une variable aléatoire réelle :math:`X` et une variable aléatoire
discrète et :math:`Y \in \acc{0,1}`, on dispose pour cela d'une liste :

.. math::

    \acc{ \pa{ X_i,Y_i } \in \R \times \acc{0,1} | 1 \infegal i \infegal N }

.. image:: rnimg/classificationnd.png

Il n'est pas facile de déterminer directement une fonction
:math:`h` qui approxime :math:`Y | X` car :math:`h` et :math:`Y`
sont toutes deux discrètes. C'est pourquoi, plutôt que de résoudre
directement ce problème, il est préférable de déterminer la
loi marginale :math:`\pr{Y=c|X} = f \pa{X,\theta,c}`.
:math:`f` est alors une fonction dont les sorties sont continues et peut
être choisie dérivable. Par exemple, :math:`f` peut être un réseau de
neurones dont les sorties vérifient :

.. math::

    f \pa{X,0} + f \pa{X,1} = \pr{0|X} + \pr{1|X} = 1

Le réseau de neurones utilisé pour cette tâche est légèrement
différent du précédent, il sera présenté ultérieurement.
Un plan a été divisé en deux demi-plan par une droite délimitant deux classes,
le réseau de neurones dont la couche cachée contient deux neurones linéaires,
a retrouvé cette séparation malgré les quelques exemples mal classés.

.. image:: rnimg/classificationnt.png

En revanche, un réseau de neurones comportant trop de coefficients
aura tendance à apprendre par coeur la classification et les quelques
erreurs de classification comme le montre la figure suivante.
La séparation produite par le réseau de neurones est de manière
évidente non linéaire puisqu'aucune droite ne peut séparer les deux classes
déterminées par cette fonction. Cette classe de modèles permet
donc de résoudre des problèmes complexes en gardant toutefois
à l'esprit, comme dans le cas de la régression, qu'il n'est pas
moins de facile de dénicher le bon modèle que dans le cas linéaire.
