
Propriétés mathématiques
========================

On s'intéresse principalement à la métrique :math:`M'` définie par
:ref:`Dynamic Minimum Keystroke <completion-metric2>` mais les résultats
seront étendues aux autres quand cela est possible.

.. contents::
    :local:

Calcul pour une complétion
++++++++++++++++++++++++++

On a besoin d'une propriété pour calculer élégamment les métriques
pour l'ensemble des complétions.

.. mathdef::
    :title: Dynamic Minimum Keystroke
    :tag: Lemme
    :lid: lemme-mks-last

    On note :math:`d(q, S)` la longueur du plus long préfixe de :math:`q` inclus dans :math:`S`.

    .. math::

        d(q, S) = \max\acc{ l(p) | p \prec q, \; p \in S, \; p \neq q}

    .. math::
        :label: lemme-m2-nlp-comp
        :nowrap:

        \begin{eqnarray*}
        M'(q, S) &=& \min_{d(q, S) \infegal k < l(q)} \acc{ M'(q[1..k], S) + \min( K(q, k, S), l(q) - k) }
        \end{eqnarray*}

Il n'est pas nécessaire de regarder tous les préfixes mais seulement ceux entre le plus long préfixe
qui est aussi une complétion et la requête :math:`q`. La démonstration est identique à la démonstration
du lemme qui suit. L'idée de cette propriété est de pouvoir réduire le coût de l'algorithme
de calcul des métriques. Ce n'est pas la seule écriture qu'on puisse en fait.

Le calcul de la métrique :math:`M'` suggère qu'on doit faire ce calcul dans le sens
des préfixes croissants mais il serait plus simple de le faire dans le sens des complétions
de poids croissant (les complétions de moindre poids sont toujours affichées avant).

.. image:: completion_img/algocomp.png

Si l'algorithme est plus simple (sens de la fléche dans le figure précédente), il faut parfois
plusieurs itérations pour obtenir la valeur finale.

Calcul pour une requête en dehors
+++++++++++++++++++++++++++++++++

Mais il est faux de dire que pour deux requêtes en dehors de l'ensemble
des complétions, :math:`q_1 \preceq q_2 \Longrightarrow M'(q_1, S) \infegal M'(q_2, S)`.
Le lemme suivant précise pourquoi

.. mathdef::
    :title: calcul de *M'(q, S)*
    :tag: Lemme
    :lid: lemme-nlp-long-completion

    On suppose que :math:`p(q, S)` est la complétion la plus longue
    de l'ensemble :math:`S` qui commence :math:`q` :

    .. math::
        :nowrap:

        \begin{eqnarray*}
        k^* &=& \max\acc{ k | q[[1..k]] \prec q \text{ et } q \in S}  \\
        p(q, S) &=& q[[1..k^*]]
        \end{eqnarray*}

    La métrique :math:`M'(q, S)` vérifie la propriété suivante :

    .. math::

        M'(q, S) = M'(p(q, S), S) + l(q) - l(p(q, S))

La métrique :math:`M'(q, S)` est égale à celle du plus long préfixe inclus
dans l'ensemble les complétions à laquelle on ajoute la différence des longueurs.
Cela correspond aux caractères que l'utilisateur doit saisir.
La démonstration est assez simple. On suppose que cela n'est pas vrai et qu'il existe
un existe :math:`k < k^*` tel que :

.. math::
    :nowrap:

    \begin{eqnarray*}
    && M'(q[[1..k]], S) + l(q) - l(q[[1..k]]) < M'(q[[1..k^*]], S) + l(q) - l(q[[1..k^*]]) \\
    & \Longrightarrow & M'(q[[1..k]], S) - k < M'(q[[1..k^*]], S) - k^* \\
    & \Longrightarrow & M'(q[[1..k]], S) + (k^* - k) < M'(q[[1..k^*]], S)
    \end{eqnarray*}

Cela signifie qu'on a réussi une façon plus efficace d'écrire le préfixe
:math:`q[[1..k^*]]`. Or par définition :math:`M'(q[[1..k^*]], S)`
est censée être le nombre de caractères minimal pour obtenir :math:`q[[1..k^*]]`.
Ce n'est donc pas possible.
Cette propriété est importante puisque pour calculer :math:`M'(q[[1..k^*]], S)`,
il suffit de regarder le plus long préfixe appartenant à l'ensemble des complétions
et seulement celui-ci. Cet algorithme et implémenté par la méthode
:meth:`enumerate_test_metric <mlstatpy.nlp.completion_simple.CompletionSystem.enumerate_test_metric>`.
En ce qui concerne la métrique :math:`M`, par définition
:math:`\forall q \notin S, \; M(q, S) = 0`. La métrique
:math:`M"` m'évoque la `côte anglaise <https://www.youtube.com/watch?v=YV54e3R-rLg>`_.
L'itération :math:`n` fonctionne de la même manière à partir du moment où
la requête considérée ne fait pas partie de l'ensemble des complétions mais
il y a l'étage d'en dessous qui pose un doute.
Il y a un brin de poésie dans ce +1. L'application de l'implémentation du calcul
de la métrique montre que :math:`M'` et :math:`M"` sont très souvent égales.
Je vais laisser ce :math:`\delta` sous forme de poésie pour le moment.

Il faudrait terminer la démonstration pour *M*...

Complétions emboîtées
+++++++++++++++++++++

On considère les complétions suivantes :

::

    actu
    actualité
    actualités
    actuel
    actuellement

Pour le préfixe *actue*, on suggère *actuel* at *actuellement*.
Pour le préfixe *actua*, on suggère *actualité* at *actualités*.
Pour le préfixe *actu*, on suggère la concaténation de ces deux listes.
Par conséquent, pour construire les listes de complétions associées à chaque préfixe,
il paraît de partir des feuilles de l'arbre puis de fusionner les listes
de complétions jusqu'au noeud racine.
Plus concrètement, si deux complétions
vérifie :math:`q_1 \preceq q_2` alors l'ensemble des complétions
vérifie :math:`C(q_1) \supset C(q_2)`. On peut même dire que :
:math:`C(q) = \cup \acc{ C(s) | s \succ q \in S}`. Cela signifie qu'une fois qu'on
a construit un trie représentant l'ensemble des complétions, il suffit de
partir des feuilles de l'arbre jusqu'à la racine pour construire la
liste des complétions à chaque étape et que pour un noeud précis,
la liste des complétions est l'union des listes de complétions des noeuds
fils.

Listes tronquées de complétions
+++++++++++++++++++++++++++++++

On reprend la première métrique :eq:`completion-metric1` qui
utilise la fonction :math:`K(q, k, S)` définie en :eq:`nlp-comp-k`.

.. math::
    :nowrap:

    \begin{eqnarray*}
    M(q, S) &=& \min_{0 \infegal k \infegal l(q)}  k + K(q, k, S)
    \end{eqnarray*}

Etant donné que le nombre minimum de caractères pour obtenir une complétion dans le trie
ne peut pas être supérieur à la longueur, si :math:`K(q, k, S) > l(q) - k`, on sait déjà que
que le préfixe :math:`q[1..k]` ne sera pas le minimum. Cette remarque est applicable
aux métriques :math:`M'` et :math:`M"`.
