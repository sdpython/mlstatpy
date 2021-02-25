
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

.. index:: Kaplan-Meier, espérance de vie

On reprend la même terminologie. A une date :math:`t_0`, on administre
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

Par simplification, on note :math:`n_i = n_{t_i}`. On suppose les :math:`t_i`
des dates à intervalles plutôt réguliers et croissants. La suite :math:`(n_i)`
est décroissantes (on ne rescuscite pas).
Ces calculs rappellent les calculs liés à l'espérance de vie
(voir `Evoluation d’une population - énoncé
<http://www.xavierdupre.fr/app/actuariat_python/helpsphinx/notebooks/seance4_projection_population_enonce.html>`_,
`Evoluation d'une population (correction)
<http://www.xavierdupre.fr/app/actuariat_python/helpsphinx/notebooks/seance4_projection_population_correction.html>`_).
L'espérance de vie est définie par :

.. math::

    \esp(D) = \sum_{i=1}^{\infty} t_i \pr{ \text{mort au temps } t_i} =
    \sum_{i=1}^{\infty} t_i \frac{n_i - n_{i+1}}{n_{i}} \prod_{j=0}^i\frac{n_j}{n_{j-1}} =
    \sum_{i=1}^{\infty} t_i \frac{n_i - n_{i+1}}{n_{i}} \frac{n_i}{n_0} =
    \sum_{i=1}^{\infty} t_i \frac{n_i - n_{i+1}}{n_0}

.. index:: fonction de survie, taux de défaillance

La courbe :math:`S(t)` est aussi appelée la fonction de survie. Si *T*
est la durée de vie d'une personne, :math:`S(t) = \pr{T > t}`.
On appelle :math:`\lambda(t)` le taux de défaillance, c'est la probabilité
que le décès survienne au temps *t* :

.. math::

    \lambda(t)dt = \pr{t \infegal T < t + dt | T \supegal T} = - \frac{S'(t)}{S(t)} dt

Régression de Cox
=================

.. index:: Cox, régression de Cox, risque de base

Le `modèle de Cox <https://fr.wikipedia.org/wiki/R%C3%A9gression_de_Cox>`_
modélise le risque de décès instantané au temps *t* selon le modèle qui suit.
Une personne est décrite par les variables :math:`X_1, ..., X_k`.

.. math::

    \lambda(t, X_1, ..., X_k) = \lambda_0(t) \exp\left(\sum_{i=1}^k \beta_i X_i\right) =
    \lambda_0(t) \exp (\beta X)

La partie :math:`\lambda_0(t)` correspond à ce qu'on observe sans
autre informations que les décès. On l'appelle aussi le *risque de base*.
C'est la probabilité moyenne
de décès instantanée. La seconde partie permet de faire varier
cette quantité selon ce qu'on sait de chaque personne.

On dit que c'est un modèle à risque proportionnel car si deux personnes sont quasiment
identiques excepté sur une variable :math:`X_i` (comme la quantité d'un poison ingérée), alors le ratio
de probabilité est :

.. math::

    \frac{\lambda(t, X_1, ..., X_i^a, ..., X_k)}{\lambda(t, X_1, ..., X_i^b, ..., X_k)} =
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

La fonction :math:`\lambda_0(t)` est en quelque sorte le taux de défaillance
moyen. On peut le calculer à partir des formules introduites au
paragraphe précédent en lissant la courbe de Kaplan-Meier avec des
splines. On peut aussi le calculer avec l'estimateur
de Breslow (voir `Analyse de survie : Méthodes non paramétriques
<http://helios.mi.parisdescartes.fr/~obouaziz/KMSurv.pdf>`_,
`Introduction à l'analyse des durées de survie
<http://www.lsta.upmc.fr/psp/Cours_Survie_1.pdf>`_).
qui repose aussi la courbe de Kaplan-Meier.

On sait que si :math:`g(t) = \log S'(t)` alors
:math:`g'(t) = \frac{S'(t)}{S(t)}`. On en déduit que :

.. math::

    \hat{\lambda_0}(t) = - \frac{d (\log(\hat{S}(t)))}{dt}

Pour la suite, on pose :math:`h(X_i, \beta) = \exp(\beta X_i)`,
et l'individu meurt au temps :math:`t_i` de l'expérience.
Une expérience est définie par la liste des couples
:math:`(X_i, t_i)`. On souhaite trouver les paramètres
:math:`\beta` qui représentent au mieux les données
de l'expérience. On définit donc :

* :math:`R_t` : l'ensemble des personnes en vie au temps *t*
* :math:`D_t` : l'ensemble qui décèdent au *t*

Par définition :math:`i \in R_{t_i}` et :math:`i \in D_{t_i}`.
On calcule le ratio :

.. math::

    Pr(\beta, t, X_i) = \frac{h(X_i, \beta) \lambda_0(t)}{\sum_{j \in R_t} h(X_j, \beta) \lambda_0(t)} =
    \frac{h(X_i, \beta) }{\sum_{j \in R_t} h(X_j, \beta) }

Pour une personne qui décède au temps *t*, ce ratio devrait être proche de 1
car on souhaite que :math:`h(X_i, \beta)` soit grand et tous les autres nuls.
On définit la vraisemblance partielle du modèle par :

.. math::

    L(\beta) = \prod_i Pr(\beta, t_i, X_i) =
    \prod_i \frac{h(X_i, \beta) }{\sum_{j \in R_{t_i}} h(X_j, \beta) }

.. index:: Breslow

Une fois qu'on a calculé les coefficients :math:`\beta` optimaux,
on peut affiner la partie :math:`\lambda_0(t)`. L'estimateur
de Breslow est :

.. math::

    \hat{B}(t) = \sum_{i | t_i \infegal t} \frac{1}{ \sum_{j \in R_{t_i}} h(\beta, X_j)}

C'est un estimateur de la fonction de survie :

.. math::

    \hat{S}(t) = \exp(-\hat{B}(t))

Notebooks
=========

.. toctree::
    :maxdepth: 1

    ../notebooks/survival

Liens, articles
===============

* `Notes de lectures <http://www.stats.ox.ac.uk/~mlunn/lecturenotes2.pdf>`_
* `On the Breslow estimator <https://dlin.web.unc.edu/wp-content/uploads/sites/1568/2013/04/Lin07.pdf>`_

Modules
=======

* `lifelines <https://lifelines.readthedocs.io/en/latest/>`_
* `scikit-survival <https://scikit-survival.readthedocs.io/en/latest/>`_
