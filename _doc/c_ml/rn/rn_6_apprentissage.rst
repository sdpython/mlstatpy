
Apprentissage d'un réseau de neurones
=====================================

.. contents::
    :local:

Le terme apprentissage est encore inspiré de la biologie et se traduit
par la minimisation de la fonction :eq:`equation_fonction_erreur_g` où
:math:`f` est un réseau de neurone défini par un :ref:`perceptron <rn_definition_perpception_1>`.
Il existe plusieurs méthodes pour effectuer celle-ci.
Chacune d'elles vise à minimiser la fonction d'erreur :

.. math::

        E\pa{W}   = G \pa{W}  =   \sum_{i=1}^{N} e\pa {Y_{i} - \widehat{Y_{i}^W}}
                                            =   \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W,X_{i}}}

Dans tous les cas, les différents apprentissages utilisent la suite
suivante :math:`\pa{ \epsilon_{t}}` vérifiant :eq:`rn_suite_epsilon_train`
et proposent une convergence vers un minimum local.

.. math::
    :label: rn_suite_epsilon_train

    \forall t>0,\quad\varepsilon_{t}\in \R_{+}^{\ast} \text{ et }
    \sum_{t\geqslant0}\varepsilon_{t}=+\infty,\quad
    \sum_{t\geqslant0}\varepsilon_{t}^{2}<+\infty

Il est souhaitable d'apprendre plusieurs fois la même fonction en modifiant
les conditions initiales de ces méthodes de manière à améliorer la robustesse de la solution.

.. _rn_apprentissage_global:

Apprentissage avec gradient global
++++++++++++++++++++++++++++++++++

L'algorithme de :ref:`rétropropagation <algo_retropropagation>` permet d'obtenir
la dérivée de l'erreur :math:`e` pour un vecteur d'entrée :math:`X`. Or l'erreur
:math:`E\pa{W}` à minimiser est la somme des erreurs pour chaque exemple
:math:`X_i`, le gradient global :math:`\partialfrac{E\pa{W}}{W}` de cette erreur
globale est la somme des gradients pour chaque exemple
(voir équation :eq:`algo_retro_1`).
Parmi les méthodes d'optimisation basées sur le gradient global, on distingue deux catégories :

* Les méthodes du premier ordre, elles sont calquées sur la
  `méthode de Newton <https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Newton>`_
  et n'utilisent que le gradient.
* Les méthodes du second ordre ou méthodes utilisant un
  `gradient conjugué <https://fr.wikipedia.org/wiki/M%C3%A9thode_du_gradient_conjugu%C3%A9>`_
  elles sont plus coûteuses en calcul mais plus performantes
  puisque elles utilisent la dérivée seconde ou une valeur approchée.

.. _rn_optim_premier_ordre:

Méthodes du premier ordre
^^^^^^^^^^^^^^^^^^^^^^^^^

Les méthodes du premier ordre sont rarement utilisées.
Elles sont toutes basées sur le principe
de la descente de gradient de Newton présentée dans
la section :ref:`optimisation_newton` :

.. mathdef::
    :title: optimisation du premier ordre
    :lid: rn_algorithme_apprentissage_1
    :tag: Algorithme

    *Initialiation*
		
    Le premier jeu de coefficients :math:`W_0` du réseau
    de neurones est choisi aléatoirement.

    .. math::

        \begin{array}{rcl}
        t   &\longleftarrow&    0 \\
        E_0 &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_0,X_{i}}}
        \end{array}

    *Calcul du gradient*

    :math:`g_t \longleftarrow \partialfrac{E_t}{W} \pa {W_t} = \sum_{i=1}^{N}
    e'\pa {Y_{i} - f \pa{W_t,X_{i}}}`

    *Mise à jour*

    .. math::

        \begin{array}{rcl}
        W_{t+1} &\longleftarrow& W_t - \epsilon_t g_t \\
        E_{t+1} &\longleftarrow& \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}} \\
        t       &\longleftarrow& t+1
        \end{array}

    *Terminaison*

    Si :math:`\frac{E_t}{E_{t-1}} \approx 1` (ou :math:`\norm{g_t} \approx 0`)
    alors l'apprentissage a convergé sinon retour au calcul du gradient.

La condition d'arrêt peut-être plus ou moins stricte selon les besoins du problème.
Cet algorithme converge vers un minimum local de la fonction d'erreur
(d'après le théorème de :ref:`convergence <theoreme_convergence>`
mais la vitesse de convergence est inconnue.

.. _rn_optim_second_ordre:

Méthodes du second ordre
^^^^^^^^^^^^^^^^^^^^^^^^

L'algorithme :ref:`apprentissage global <rn_apprentissage_global>` fournit le canevas des
méthodes d'optimisation du second ordre. La mise à jour des coefficients est différente car
elle prend en compte les dernières valeurs des coefficients ainsi que les
derniers gradients calculés. Ce passé va être utilisé pour estimer une
direction de recherche pour le minimum différente de celle du gradient,
cette direction est appelée gradient conjugué (voir [Moré1977]_).

Ces techniques sont basées sur une approximation du second degré de la fonction à minimiser.
On note :math:`M` le nombre de coefficients du réseau de neurones (biais compris).
Soit :math:`h: \R^{M} \dans \R` la fonction d'erreur associée au réseau de neurones :
:math:`h \pa {W} = \sum_{i} e \pa{Y_i,f \pa{ W,X_i} }`.
Au voisinage de :math:`W_{0}`, un développement limité donne :

.. math::

    h \pa {W}     =   h\pa {W_0}  + \frac{\partial h\left( W_{0}\right)  }{\partial W}\left( W-W_{0}\right) +\left(
    W-W_{0}\right) ^{\prime}\frac{\partial^{2}h\left(  W_{0}\right)  }{\partial W^{2}}\left( W-W_{0}\right) +o\left\|
    W-W_{0}\right\|  ^{2}

Par conséquent, sur un voisinage de :math:`W_{0}`, la fonction :math:`h\left( W\right)`
admet un minimum local si :math:`\frac{\partial^{2}h\left( W_{0}\right) }{\partial W^{2}}`
est définie positive strictement.

*Rappel :* :math:`\dfrac{\partial^{2}h\left(  W_{0}\right)  }{\partial W^{2}}`
est définie positive strictement :math:`\Longleftrightarrow\forall Z\in\R^{N},\; Z\neq0\Longrightarrow
Z^{\prime}\dfrac{\partial ^{2}h\left( W_{0}\right)  }{\partial W^{2}}Z>0`.

Une matrice symétrique définie strictement positive est inversible,
et le minimum est atteint pour la valeur :

.. math::
    :nowrap:
    :label: rn_hessien

    \begin{eqnarray}
    W_{\min}= W_0 + \frac{1}{2}\left[  \dfrac{\partial^{2}h\left(  W_{0}\right) }
    		{\partial W^{2}}\right]  ^{-1}\left[  \frac{\partial h\left(  W_{0}\right)
    }{\partial W}\right] \nonumber
    \end{eqnarray}

Néanmoins, pour un réseau de neurones, le calcul de la dérivée seconde est coûteux,
son inversion également. C'est pourquoi les dernières valeurs des coefficients
et du gradient sont utilisées afin d'approcher cette dérivée seconde ou directement
son inverse. Deux méthodes d'approximation sont présentées :

* L'algorithme `BFGS (Broyden-Fletcher-Goldfarb-Shano) <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_
  ([Broyden1967]_, [Fletcher1993]_), voir aussi les versions `L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_.
* L'algoritmhe `DFP  (Davidon-Fletcher-Powell) <https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula>`_
  ([Davidon1959]_, [Fletcher1963]_).

La figure du :ref:`gradient conjugué <figure_gradient_conjugue>` est couramment employée
pour illustrer l'intérêt des méthodes de gradient conjugué.
Le problème consiste à trouver le minimum d'une fonction quadratique,
par exemple, :math:`G\pa{x,y} = 3x^2 + y^2`. Tandis que le gradient est orthogonal
aux lignes de niveaux de la fonction :math:`G`, le gradient conjugué se dirige plus
sûrement vers le minimum global.

.. mathdef::
    :title: Gradient conjugué
    :lid: figure_gradient_conjugue
    :tag: Figure

    .. image:: rnimg/Conjugate_gradient_illustration.png
        :alt: Wikipedia

    Gradient et gradient conjugué sur une ligne de niveau de la fonction :math:`G\pa{x,y} = 3x^2 + y^2`,
    le gradient est orthogonal aux lignes de niveaux de la fonction :math:`G`,
    mais cette direction est rarement la bonne à moins que le point
    :math:`\pa{x,y}` se situe sur un des axes des ellipses,
    le gradient conjugué agrège les derniers déplacements et propose une direction
    de recherche plus plausible pour le minimum de la fonction.
    Voir `Conjugate Gradient Method <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.

Ces méthodes proposent une estimation de la dérivée seconde
(ou de son inverse) utilisée en :eq:`rn_hessien`.
Dans les méthodes du premier ordre, une itération permet de calculer les
poids :math:`W_{t+1}` à partir des poids :math:`W_t` et du
gradient :math:`G_t`. Si ce gradient est petit, on peut supposer
que :math:`G_{t+1}` est presque égal au produit de la dérivée seconde par
:math:`G_t`. Cette relation est mise à profit pour construire une estimation
de la dérivée seconde. Cette matrice notée :math:`B_t` dans
l'algorithme :ref:`BFGS <rn_algo_bfgs>`
est d'abord supposée égale à l'identité puis actualisée à chaque
itération en tenant de l'information apportée par chaque déplacement.

.. mathdef::
    :title: BFGS
    :tag: Algorithme
    :lid: rn_algo_bfgs

    Le nombre de paramètres de la fonction :math:`f` est :math:`M`.

    *Initialisation*

    Le premier jeu de coefficients :math:`W_0` du réseau de neurones est
    choisi aléatoirement.

    .. math::

        \begin{array}{lcl}
        t   &\longleftarrow&    0 \\
        E_0 &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_0,X_{i}}} \\
        B_0 &\longleftarrow&    I_M \\
        i   &\longleftarrow&    0
        \end{array}

    *Calcul du gradient*

    .. math::

        \begin{array}{lcl}
        g_t &\longleftarrow& \partialfrac{E_t}{W} \pa {W_t}= \sum_{i=1}^{N} e'\pa {Y_{i} - f \pa{W_t,X_{i}}} \\
        c_t &\longleftarrow& B_t g_t
        \end{array}

    *Mise à jour des coefficients*

    .. math::

        \begin{array}{lcl}
        \epsilon^*  &\longleftarrow&    \underset{\epsilon}{\arg \inf} \; \sum_{i=1}^{N}
                 e\pa {Y_i - f \pa{W_t - \epsilon c_t,X_i}}  \\
        W_{t+1}     &\longleftarrow&    W_t - \epsilon^* c_t \\
        E_{t+1}     &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}} \\
        t           &\longleftarrow&    t+1
        \end{array}

    *Mise à jour de la marice :math:`B_t`*

    | si :math:`t - i \supegal M` ou :math:`g'_{t-1} B_{t-1} g_{t-1} \infegal 0` ou :math:`g'_{t-1} B_{t-1} \pa {g_t - g_{t-1}} \infegal 0`
    |   :math:`B_{t} \longleftarrow I_M`
    |   :math:`i \longleftarrow  t`
    | sinon
    |   :math:`s_t \longleftarrow    W_t - W_{t-1}`
    |   :math:`d_t    \longleftarrow    g_t - g_{t-1}`
    |   :math:`B_{t}  \longleftarrow    B_{t-1} +   \pa{1 + \dfrac{ d'_t B_{t-1} d_t}{d'_t s_t}}\dfrac{s_t s'_t} {s'_t d_t}- \dfrac{s_t d'_t B_{t-1} +  B_{t-1} d_t s'_t } { d'_t s_t }`

    *Terminaison*

    Si :math:`\frac{E_t}{E_{t-1}} \approx 1` alors l'apprentissage a convergé sinon retour au calcul
    du gradient.

Lorsque la matrice :math:`B_t` est égale à l'identité,
le gradient conjugué est égal au gradient. Au fur et
à mesure des itérations, cette matrice toujours
symétrique évolue en améliorant la convergence de l'optimisation.
Néanmoins, la matrice :math:`B_t` doit être "nettoyée"
(égale à l'identité) fréquemment afin d'éviter qu'elle
n'agrège un passé trop lointain. Elle est aussi nettoyée lorsque
le gradient conjugué semble trop s'éloigner du véritable gradient
et devient plus proche d'une direction perpendiculaire.

La convergence de cet algorithme dans le cas des réseaux de
neurones est plus rapide qu'un algorithme du premier ordre,
une preuve en est donnée dans [Driancourt1996]_.

En pratique, la recherche de :math:`\epsilon^*` est réduite car
le calcul de l'erreur est souvent coûteux, il peut être effectué
sur un grand nombre d'exemples. C'est pourquoi on remplace
l'étape de mise à jour de l'algorithme :ref:`BFGS <rn_algo_bfgs>`
par celle-ci :

.. mathdef::
    :title: BFGS'
    :lid: rn_algo_bfgs_prime
    :tag: Algorithme

    Le nombre de paramètre de la fonction :math:`f` est :math:`M`.

    *Initialisation, calcul du gradient*

    Voir :ref:`BFGS <rn_algo_bfgs>`.

    *Recherche de :math:`\epsilon^*`*

    | :math:`\epsilon^*  \longleftarrow    \epsilon_0`
    | while :math:`E_{t+1} \supegal E_t` et :math:`\epsilon^* \gg 0`
    |   :math:`\epsilon^*  \longleftarrow   \frac{\epsilon^*}{2}`
    |   :math:`W_{t+1}     \longleftarrow    W_t - \epsilon^* c_t`
    |   :math:`E_{t+1}     \longleftarrow    \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}}`
    |
    | if :math:`\epsilon_* \approx 0` et :math:`B_t \neq I_M`
    |   :math:`B_{t}       \longleftarrow   I_M`
    |   :math:`i           \longleftarrow    t`
    |   Retour au calcul du gradient.

    *Mise à jour des coefficients*

    .. math::

        \begin{array}{lcl}
        W_{t+1}     &\longleftarrow&    W_t - \epsilon^* c_t \\
        E_{t+1}     &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}} \\
        t           &\longleftarrow&    t+1
        \end{array}

    *Mise à jour de la matrice :math:`B_t`, temrinaison*

    Voir :ref:`BFGS <rn_algo_bfgs>`.

		
L'algorithme DFP est aussi un algorithme de gradient conjugué
qui propose une approximation différente de l'inverse de la dérivée seconde.
		
.. mathdef::
    :title: DFP
    :lid: rn_algo_dfp
    :tag: Algorithme

    Le nombre de paramètre de la fonction :math:`f` est :math:`M`.
		
    *Initialisation*

    Le premier jeu de coefficients :math:`W_0`
    du réseau de neurones est choisi aléatoirement.

    .. math::

        \begin{array}{lcl}
        t   &\longleftarrow&    0 \\
        E_0 &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_0,X_{i}}} \\
        B_0 &\longleftarrow&    I_M \\
        i   &\longleftarrow&    0
        \end{array}

    *Calcul du gradient*

    .. math::

        \begin{array}{lcl}
        g_t &\longleftarrow& \partialfrac{E_t}{W} \pa {W_t}= \sum_{i=1}^{N} e'\pa {Y_{i} - f \pa{W_t,X_{i}}} \\
        c_t &\longleftarrow& B_t g_t
        \end{array}

    *Mise à jour des coefficients*

    .. math::

        \begin{array}{lcl}
        \epsilon^*  &\longleftarrow&    \underset{\epsilon}{\arg \inf} \;
                                     \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_t - \epsilon c_t,X_i}}  \\
        W_{t+1}     &\longleftarrow&    W_t - \epsilon^* c_t \\
        E_{t+1}     &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}} \\
        t           &\longleftarrow&    t+1
        \end{array}

    *Mise à jour de la matrice :math:`B_t`*

    | si :math:`t - i \supegal M` ou :math:`g'_{t-1} B_{t-1} g_{t-1} \infegal 0` ou :math:`g'_{t-1} B_{t-1} \pa {g_t - g_{t-1}} \infegal 0`
    |   :math:`B_{t}       \longleftarrow    I_M`
    |   :math:`i           \longleftarrow    t`
    | sinon
    |   :math:`d_t         \longleftarrow    W_t - W_{t-1}`
    |   :math:`s_t         \longleftarrow    g_t - g_{t-1}`
    |   :math:`B_{t}       \longleftarrow`    B_{t-1} +     \dfrac{d_t d'_t} {d'_t s_t} - \dfrac{B_{t-1} s_t s'_t B_{t-1} } { s'_t B_{t-1} s_t }`

    *Terminaison*

    Si :math:`\frac{E_t}{E_{t-1}} \approx 1` alors l'apprentissage a convergé sinon retour à
    du calcul du gradient.

Seule l'étape de mise à jour :math:`B_t` diffère dans les
algorithmes :ref:`BFGS <rn_algo_bfgs>` et :ref:`DFP <rn_algo_dfp>`.
Comme l'algorithme :ref:`BFGS <rn_algo_bfgs>`,
on peut construire une version :ref:`DFP <rn_algo_dfp>`'
inspirée de l'algorithme :ref:`BFGS' <rn_algo_bfgs_prime>`.

Apprentissage avec gradient stochastique
++++++++++++++++++++++++++++++++++++++++

Compte tenu des courbes d'erreurs très :ref:`accidentées <figure_courbe_accident>`
dessinées par les réseaux de neurones, il existe une multitude de minima
locaux. De ce fait, l'apprentissage global converge rarement vers le
minimum global de la fonction d'erreur lorsqu'on applique les algorithmes
basés sur le gradient global. L'apprentissage avec gradient stochastique
est une solution permettant de mieux explorer ces courbes d'erreurs.
De plus, les méthodes de gradient conjugué nécessite le stockage d'une
matrice trop grande parfois pour des fonctions ayant quelques milliers
de paramètres. C'est pourquoi l'apprentissage avec gradient stochastique
est souvent préféré à l'apprentissage global pour de grands réseaux de
neurones alors que les méthodes du second ordre trop coûteuses en
calcul sont cantonnées à de petits réseaux. En contrepartie, la
convergence est plus lente. La démonstration de cette convergence nécessite
l'utilisation de quasi-martingales et est une convergence presque sûre [Bottou1991]_.

.. mathdef::
    :title: Exemple de minimal locaux
    :tag: Figure
    :lid: figure_courbe_accident

    .. image:: rnimg/errminloc.png

.. mathdef::
    :title: apprentissage stochastique
    :tag: Algprithme
    :lid: rn_algorithme_apprentissage_2

    *Initialisation*

    Le premier jeu de coefficients :math:`W_0`
    du réseau de neurones est choisi aléatoirement.

    .. math::

        \begin{array}{lcl}
        t       &\longleftarrow&    0 \\
        E_0 &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_0,X_{i}}}
        \end{array}

    *Récurrence*

    | :math:`W_{t,0} \longleftarrow    W_0`
    | for :math:`t'` in :math:`0..N-1`
    |   :math:`i \longleftarrow` nombre aléatoire dans :math:`\ensemble{1}{N}`
    |   :math:`g \longleftarrow \partialfrac{E}{W} \pa {W_{t,t'}}=  e'\pa {Y_{i} - f\pa{W_{t,t'},X_{i}}}`
    |   :math:`W_{t,t'+1} \longleftarrow    W_{t,t'} - \epsilon_t g`
    | :math:`W_{t+1} \longleftarrow W_{t,N}`
    | :math:`E_{t+1} \longleftarrow \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_{t+1},X_{i}}}`
    | :math:`t \longleftarrow t+1`

    *Terminaison*
		
    Si :math:`\frac{E_t}{E_{t-1}} \approx 1`
    alors l'apprentissage a convergé sinon retour au
    calcul du gradient.
		

En pratique, il est utile de converser le meilleur jeu de
coefficients : :math:`W^* = \underset{u \supegal 0}{\arg \min} \; E_{u}`
car la suite :math:`\pa {E_u}_{u \supegal 0}` n'est pas une suite décroissante.
