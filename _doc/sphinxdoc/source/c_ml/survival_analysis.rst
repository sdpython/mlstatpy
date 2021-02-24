
.. _l-survival-analysis:

=================
Analyse de survie
=================

.. index:: analyse de survie

L'`analyse de survie
<https://fr.wikipedia.org/wiki/Analyse_de_survie>`_
est un sujet qu'on commence à voir
poindre en assurance et plus généralement en assurance.
C'est domaine développé
pour mesurer les effets d'une substance, d'un médicament
sur un corps vivant, une personne.

.. contents::
    :local:

Lien avec le machine learning
=============================

En assurance, on cherche souvent à prédire si une personne aura
un accident ou pas. Pour cela, il faut avoir des données,
une base de données dans laquelle sont enregistrés des accidents.
L'accident en question peut avoir lieu au début du contrat, quelques
années plus tard ou jamais. Lorsqu'aucun accident n'est associé
à une personne, il se peut qu'il ne se produise aucun accident
ou que celui-ci ne s'est pas encore produit. Modéliser ce problème
de prédiction permet d'introduire le temps et prendre en compte
le fait que les données sont tronquées : on ne sait pour une personne
que si un accident s'est produit ou pas entre le début du contrat
et aujourd'hui.

Courbe de Kaplan-Meier
======================

On reprend la même terminologie. A une date *t*, on administre
un traitement à une personne, un animal, une plante. Cet être vivant
meurt à un temps *t + d*. Le traitement a-t-il amélioré sa survie ?
On considère deux temps :math:`t_1` et :math:`t_2`, la probabilité
de décès entre ces deux temps peut être estimé par
:math:`\frac{n_{t_2} - n_{t_1}}{n_{t_1}}` où :math:`n_{t_i}` est la
population vivante au temps :math:`t_i` (depuis le début du traitement).

On en déduit la probabilité de rester vivant jusqu'au temps :math:`t_i`
qui est l'estimateur de `Kaplan-Meier
<https://fr.wikipedia.org/wiki/Estimateur_de_Kaplan-Meier>`_
:math:`\hat{S}(t_i)` :

.. math::

    \begin{array}{rcl}
    \hat{S}(t_i) &=& \prod_{i=1}^i \left( 1 - \frac{n_{t_{i-1}} - n_{t_{i}}}{n_{t_{i-1}}} \right) \\
    &=& \prod_{i=1}^i \frac{n_{t_i}}{n_{t_{i-1}}} = \prod_{i=1}^i \frac{n_i}{n_{i-1}}
    \end{array}

Ces calculs rappellent les calculs liés à l'espérance de vie
(voir `Evoluation d’une population - énoncé
<http://www.xavierdupre.fr/app/actuariat_python/helpsphinx/notebooks/seance4_projection_population_enonce.html>`_,
`Evoluation d'une population (correction)
<http://www.xavierdupre.fr/app/actuariat_python/helpsphinx/notebooks/seance4_projection_population_correction.html>`_).

Régression de Cox
=================

Le `modèle de Cox <https://fr.wikipedia.org/wiki/R%C3%A9gression_de_Cox>`_
modélise le risque de décès instantané au temps *t* selon le modèle qui suit.
Une personne est décrite par les variables :math:`X_1, ..., X_n`.

.. math::

    \lambda(t, X_1, ..., X_n) = \lambda_0(t) \exp\left(\sum_{i=1}^n \beta_i X_i\right)

On dit que c'est un modèle à risque proportionnel car si deux personnes sont quasiment
identiques excepté sur une variable :math:`X_i` (comme la quantité d'un poison ingérée), alors le ratio
de probabilité est :

.. math::

    \frac{\lambda(t, X_1, ..., X_i^a, ..., X_n)}{\lambda(t, X_1, ..., X_i^b, ..., X_n)} =
    \frac{\exp(\beta_i X_i^a)} {\exp(\beta_i X_i^b)} =
    \exp\left(\beta_i (X_i^a - X_i^b)\right)

L'hypothèse des risques proportionnel est en quelque sorte intuitive.
Plus on ingère un poison, plus on a de chances d'en subir les conséquences.
Mais ce n'est pas toujours le cas, le documentaire
`La fabrique de l'ignorance
<https://www.arte.tv/fr/videos/091148-000-A/la-fabrique-de-l-ignorance/>`_
revient sur les effets du `bisphénol A <https://fr.wikipedia.org/wiki/Bisph%C3%A9nol_A>`_
qui serait déjà pertubateur à très petite dose. Il ne prend pas en compte
les effets croisés non plus (voir `Les perturbateurs endocriniens Comprendre où en est la recherche
<https://hal-anses.archives-ouvertes.fr/anses-02289024/document>`_).

La fonction :math:`\lambda_0(t)` est souvent approchée avec l'estimateur
de Breslow (voir `Analyse de survie : Méthodes non paramétriques
<http://helios.mi.parisdescartes.fr/~obouaziz/KMSurv.pdf>`_,
`Introduction à l'analyse des durées de survie
<http://www.lsta.upmc.fr/psp/Cours_Survie_1.pdf>`_).
La fonction :math:`\lambda_0` est calculé à partir de la courbe de
Kaplan-Meier.

.. math::

    \hat{\lambda_0}(t) = − log(\hat{S}(t))

Notebooks
=========

.. toctree::
    :maxdepth: 1

    ../notebooks/survival
