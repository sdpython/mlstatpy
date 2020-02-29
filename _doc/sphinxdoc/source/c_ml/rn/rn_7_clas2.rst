
.. _rn-classification:

Classification
==============

.. contents::
    :local:

Vraisemblance d'un échantillon de variable suivant une loi multinomiale
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Soit :math:`\pa{Y_i}_{1 \infegal i \infegal N}`
un échantillon de variables aléatoires i.i.d. suivant la loi multinomiale
:math:`\loimultinomiale { \vecteurno{p_1}{p_C}}`.
On définit :math:`\forall k \in \intervalle{1}{C}, \; d_k = \frac{1}{N}
\sum_{i=1}^{N} \indicatrice{Y_i = k}`.
La vraisemblance de l'échantillon est :

.. math::
    :nowrap:
    :label: rn_equation_vraisemblance_kullbck_leiber

    \begin{eqnarray}
    L\pa{\vecteurno{Y_1}{Y_N}, \vecteurno{p_1}{p_C}} &=& \prod_{i=1}^{n} p_{Y_i} \nonumber\\
    \ln L\pa{\vecteurno{Y_1}{Y_N}, \vecteurno{p_1}{p_C}} &=& \sum_{i=1}^{n} \ln p_{Y_i}  \nonumber\\
    \ln L\pa{\vecteurno{Y_1}{Y_N}, \vecteurno{p_1}{p_C}} &=& \sum_{k=1}^{C} \cro{ \pa{\ln p_k}
                                                                    \sum_{i=1}^{N}  \indicatrice{Y_i = k}}  \nonumber\\
    \ln L\pa{\vecteurno{Y_1}{Y_N}, \vecteurno{p_1}{p_C}} &=& N \sum_{k=1}^{C} d_k \ln p_k
                    \nonumber
    \end{eqnarray}

Cette fonction est aussi appelée distance de
`Kullback-Leiber <https://fr.wikipedia.org/wiki/Divergence_de_Kullback-Leibler>`_
([Kullback1951]_), elle mesure la distance entre deux
distributions de variables aléatoires discrètes.
L'`estimateur de maximum de vraisemblance <https://fr.wikipedia.org/wiki/Maximum_de_vraisemblance>`_ (emv)
est la solution du problème suivant :

.. mathdef::
    :title: estimateur du maximum de vraisemblance
    :lid: problem_emv
    :tag: Problème

    Soit un vecteur :math:`\vecteur{d_1}{d_N}` tel que :

    .. math::

        \left\{
        \begin{array}{l}
        \sum_{k=1}^{N} d_k = 1 \\
        \forall k \in \ensemble{1}{N}, \; d_k \supegal 0
        \end{array}
        \right.

    On cherche le vecteur :math:`\vecteur{p_1^*}{p_N^*}` vérifiant :

    .. math::

        \begin{array}{l}
        \vecteur{p_1^*}{p_N^*} = \underset{ \vecteur{p_1}{p_C} \in \R^C }{\arg \max}
                       \sum_{k=1}^{C} d_k \ln p_k \medskip \\
        \quad \text{avec } \left \{
            \begin{array}{l}
            \forall k \in \intervalle{1}{C}, \; p_k \supegal 0 \\
            \text{et } \sum_{k=1}^{C} p_k = 1
            \end{array}
            \right.
        \end{array}

.. mathdef::
    :title: résolution du problème du maximum de vraisemblance
    :lid: theorem_problem_emv
    :tag: Théorème

    La solution du problème du :ref:`maximum de vraisemblance <problem_emv>`
    est le vecteur :

    .. math::

        \vecteur{p_1^*}{p_N^*} = \vecteur{d_1}{d_N}

*Démonstration*

Soit un vecteur :math:`\vecteur{p_1}{p_N}` vérifiant les conditions :

.. math::

    \left\{
    \begin{array}{l}
    \sum_{k=1}^{N} p_k = 1 \\
    \forall k \in \ensemble{1}{N}, \;  p_k \supegal 0
    \end{array}
    \right.

La fonction :math:`x \longrightarrow \ln x` est concave, d'où :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \Delta  &=&         \sum_{k=1}^{C} d_k \ln p_k - \sum_{k=1}^{C} d_k \ln d_k \\
            &=&         \sum_{k=1}^{C} d_k \pa{ \ln p_k - \ln d_k } = \sum_{k=1}^{C} d_k \ln \frac{p_k}{d_k} \\
            &\infegal&  \ln \pa{ \sum_{k=1}^{C} d_k \frac{p_k}{d_k} } = \ln \pa { \sum_{k=1}^{C} p_k } = \ln 1 = 0 \\
            &\infegal&  0
    \end{eqnarray*}

La distance de KullBack-Leiber compare deux distributions de
probabilités entre elles. C'est elle qui va faire le
lien entre le problème de :ref:`classification discret <probleme_classification>`
et les réseaux de neurones pour lesquels il faut impérativement une fonction d'erreur dérivable.

.. _subsection_classifieur:

Problème de classification pour les réseaux de neurones
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Le problème de :ref:`classification <probleme_classification>`
est un cas particulier de celui qui suit pour lequel il
n'est pas nécessaire de connaître la classe d'appartenance
de chaque exemple mais seulement les probabilités d'appartenance
de cet exemple à chacune des classes.

Soient une variable aléatoire continue :math:`X \in \R^p`
et une variable aléatoire discrète multinomiale
:math:`Y \in \intervalle{1}{C}`, on veut estimer la loi de :

.. math::

    Y|X \sim \loimultinomiale {p_1\pa{W,X},\dots , p_C\pa{W,X}}
    \text { avec } W \in \R^M

Le vecteur :math:`\vecteur{p_1\pa{W,X}}{p_C\pa{W,X}}`
est une fonction :math:`f` de :math:`\pa{W,X}` où
:math:`W` est l'ensemble des :math:`M` paramètres du modèle.
Cette fonction possède :math:`p` entrées et :math:`C` sorties.
Comme pour le problème de la régression, on cherche les
poids :math:`W` qui correspondent le mieux à l'échantillon :

.. math::

    A = \acc{\left. \pa {X_i,y_i=\pa{\eta_i^k}_{1 \infegal k \infegal C}} \in \R^p \times \cro{0,1}^C
               \text{ tel que } \sum_{k=1}^{c}y_i^k=1 \right| 1 \infegal i \infegal N }

On suppose que les variables :math:`\pa{Y_i|X_i}_{1 \infegal i \infegal N}`
suivent les lois respectives :math:`\pa{\loimultinomiale{y_i}}_{1 \infegal i \infegal N}`
et sont indépendantes entre elles, la vraisemblance du modèle
vérifie d'après l'équation :eq:`rn_equation_vraisemblance_kullbck_leiber` :

.. math::
    :nowrap:

    \begin{eqnarray*}
    L_W & \propto & \prod_{i=1}^{N}\prod_{k=1}^{C} \cro{p_k \pa{W,X_i}}^{\pr{Y_i=k}} \\
    \ln L_W & \propto & \sum_{i=1}^{N}\sum_{k=1}^{C} \eta_i^k \ln\cro { p_k\pa{W,X_i}}
    \end{eqnarray*}

La solution du problème  :math:`\overset{*}{W} = \underset{W \in \R^l}{\arg \max} \; L_W`
est celle d'un problème d'optimisation sous contrainte. Afin de contourner
ce problème, on définit la fonction :math:`f` :

.. math::

    \begin{array}{l}
    f : \R^M \times \R^p \longrightarrow \R^C \\
    \forall \pa{W,x} \in \R^M \times \R^p, \; f\pa{W,x} = \pa{f_1\pa{W,x}}, \dots ,
                    f_C\pa{W,x} \vspace{0.5ex}\\
    \text{et }\forall i \in \intervalle{1}{N}, \; \forall k \in \intervalle{1}{C}, \;
    				p^k \pa{W,X_i} = \dfrac{e^{f_k\pa{W,X_i}}}
    {\sum_{l=1}^{C}e^{f_l\pa{W,X_i}}}
    \end{array}

Les contraintes sur :math:`\pa{p^k\pa{W,X_i}}` sont bien vérifiées :

.. math::

    \begin{array}{l}
    \forall i \in \intervalle{1}{N},\; \forall k \in \intervalle{1}{C}, \; p^k\pa{W,X_i} \supegal 0 \\
    \forall i \in \intervalle{1}{N},\; \sum_{k=1}^{C} p^k\pa{W,X_i} = 1
    \end{array}

On en déduit que :

.. math::
    :nowrap:

		\begin{eqnarray*}
		\ln L_W & \propto & \sum_{i=1}^{N}\sum_{k=1}^{C} \; \eta_i^k  \cro{ f_k\pa{W,X_i} - \ln
		\cro{\sum_{l=1}^{C}e^{f_l\pa{W,X_i}}}} \\
		\ln L_W & \propto & \sum_{i=1}^{N}\sum_{k=1}^{C} \; \eta_i^k  f_k\pa{W,X_i} -
		                  \sum_{i=1}^{N}  \ln \cro{\sum_{l=1}^{C}e^{f_l\pa{W,X_i}}}
		                  \underset{=1}{\underbrace{\sum_{k=1}^{C} \eta_i^k}}
		\end{eqnarray*}

D'où :

.. math::
    :nowrap:
    :label: nn_classification_vraisemblance_error

    \begin{eqnarray}
        \begin{array}[c]{c}
        \ln L_W \propto  \sum_{i=1}^{N} \sum_{k=1}^{C} \eta_i^k  f_k\pa{W,X_i} - \sum_{i=1}^{N}
         \ln \cro{ \sum_{l=1}^{C} e^{f_l\pa{W,X_i} }}
        \end{array} \nonumber
    \end{eqnarray}

Ceci mène à la définition du problème de classification suivant :

.. mathdef::
    :tag: Problème
    :title: classification
    :lid: problem_classification_2

    Soit :math:`A` l'échantillon suivant :

    .. math::

        A = \acc {\left. \pa {X_i,y_i=\pa{\eta_i^k}_{1 \infegal k \infegal C}} \in
                                                \R^p \times \R^C
                            \text{ tel que } \sum_{k=1}^{c}\eta_i^k=1 \right| 1 \infegal i \infegal N }

    :math:`y_i^k` représente la probabilité que l'élément
    :math:`X_i` appartiennent à la classe :math:`k` :
    :math:`\eta_i^k = \pr{Y_i = k | X_i}`

    Le classifieur cherché est une fonction :math:`f` définie par :

    .. math::

        \begin{array}{rcl}
        f : \R^M \times \R^p &\longrightarrow& \R^C \\
        \pa{W,X}    &\longrightarrow&  \vecteur{f_1\pa{W,X}}{f_p\pa{W,X}} \\
        \end{array}

    Dont le vecteur de poids :math:`W^*` est égal à :

    .. math::

        W^* =   \underset{W}{\arg \max} \;
                \sum_{i=1}^{N} \sum_{k=1}^{C} \eta_i^k  f_k\pa{W,X_i} -
                \sum_{i=1}^{N}  \ln \cro{ \sum_{l=1}^{C} e^{f_l\pa{W,X_i} }}

Réseau de neurones adéquat
++++++++++++++++++++++++++

Dans le problème précédent, la maximisation de
:math:`\overset{*}{W} = \underset{W \in \R^M}{\arg \max} \, L_W`
aboutit au choix d'une fonction :

.. math::

    X \in \R^p \longrightarrow f(\overset{*}{W},X) \in \R^C

Le réseau de neurones :ref:`suivant <figure_rn_classification_adequat_figure>`
:math:`g : \pa{W,X} \in \R^M \times \R^p \longrightarrow \R^C`
choisi pour modéliser :math:`f` aura pour sorties :

.. math::

    \begin{array}{l}
    X \in \R^p \longrightarrow g(\overset{*}{W},X) \in \R^C\\
    \forall k \in \intervalle{1}{C}, \; g_k \pa{W,X} = e^{f_k\pa{W,X}}
    \end{array}

.. mathdef::
    :title: Réseau de neurones adéquat pour la classification
    :lid: figure_rn_classification_adequat_figure
    :tag: Figure

    .. image:: rnimg/rn_clad.png

On en déduit que la fonction de transert des neurones de la couche de sortie est :
:math:`x \longrightarrow e^x`.
La probabilité pour le vecteur :math:`x\in\R^p`
d'appartenir à la classe :math:`k\in\intervalle{1}{C}` est
:math:`p_k(\overset{*}{W},x) = \pr{Y=k|x} = \dfrac { g_k(\overset{*}{W},x)}
{\sum_{l=1}^{C} g_l(\overset{*}{W},x) }`.
La fonction d'erreur à minimiser est l'opposé de la log-vraisemblance du modèle :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \overset{*}{W} &=& \underset{W \in \R^M}{\arg \min}
          \cro {\sum_{i=1}^{N} \pa { - \sum_{k=1}^{C} \eta_i^k  \ln \pa{g_k\pa{W,X_i}} +
                        \ln \cro{ \sum_{l=1}^{C} g_l\pa{W,X_i} }}} \\
          &=& \underset{W \in \R^M}{\arg \min}  \cro {\sum_{i=1}^{N} h\pa{W,X_i,\eta_i^k}}
    \end{eqnarray*}

On note :math:`C_{rn}` le nombre de couches du réseau de neurones,
:math:`z_{C_{rn}}^k` est la sortie :math:`k` avec
:math:`k \in \intervalle{1}{C}`,
:math:`g_k\pa{W,x} = z_{C_{rn}}^k = e^{y_{C_{rn}}^k}` où
:math:`y_{C_{rn}}^k` est le potentiel du neurone :math:`k` de la couche de sortie.

On calcule :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \partialfrac{h\pa{W,X_i,y_i^k}}{y_{C_{rn}}^k} &=& - \eta_i^k +  \dfrac{z_{C{rn}}^i}{\sum_{m=1}^{C}z_{C{rn}}^m} \\
    &=& p_k(\overset{*}{W},x) - \eta_i^k
    \end{eqnarray*}

Cette équation permet d'adapter l'algorithme de la :ref:`rétropropagation <algo_retropropagation>`
décrivant rétropropagation pour le problème de la classification et pour
un exemple :math:`\pa {X,y=\pa{\eta^k}_{1 \infegal k \infegal C}}`.
Seule la couche de sortie change.

.. mathdef::
    :title: rétropropagation
    :lid: algo_retropropagation_class
    :tag: Algorithme

    Cet algorithme de rétropropagation est l'adaptation de
    :ref:`rétropropagation <algo_retropropagation>` pour le problème
    de la classification. Il suppose que l'algorithme de :ref:`propagation <algo_propagation>`
    a été préalablement exécuté.
    On note :math:`y'_{c,i} = \partialfrac{e}{y_{c,i}}`,
    :math:`w'_{c,i,j} = \partialfrac{e}{w_{c,i,j}}` et
    :math:`b'_{c,i} = \partialfrac{e}{b_{c,i}}`.

    *Initialiasation*

    | for :math:`i` in :math:`1..C_C`
    |   :math:`y'_{C,i} \longleftarrow \dfrac{z_{C,i}} {\sum_{l=1}^{C} z_{C,l} } - \eta_i`

    *Récurrence, Terminaison*

    Voir :ref:`rétropropagation <algo_retropropagation>`.

On vérifie que le gradient s'annule lorsque le réseau de neurones
retourne pour l'exemple :math:`\pa{X_i,y_i}` la
distribution de :math:`Y|X_i \sim \loimultinomiale{y_i}`.
Cet algorithme de rétropropagation utilise un vecteur désiré de
probabilités :math:`\vecteur{\eta_1}{\eta_{C_C}}` vérifiant
:math:`\sum_{i=1}^{C_C} \, \eta_i = 1`.
L'expérience montre qu'il est préférable d'utiliser un vecteur vérifiant la contrainte :

.. math::
    :nowrap:

    \begin{eqnarray}
    && \forall i \in \ensemble{1}{C_C}, \;  \min\acc{ \eta_i, 1-\eta_i} > \alpha \nonumber \\
    && \text{avec } \alpha > 0 \nonumber
    \end{eqnarray}

Généralement, :math:`\alpha` est de l'ordre de :math:`0,1` ou
:math:`0,01`. Cette contrainte facilite le calcul de la vraisemblance
et évite l'obtention de gradients quasi-nuls qui freinent l'apprentissage
lorsque les fonctions exponnetielles sont saturées (voir [Bishop1995]_).
