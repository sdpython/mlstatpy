
Problème d'optimisation
=======================

.. contents::
    :local:

Enoncé 1
++++++++

.. mathdef::
    :title: Optimiser un système de complétion
    :lid: optim-nlp-comp
    :tag: Problème

    On suppose que l'ensemble des complétions :math:`C=\acc{c_j}` est connu.
    On souhaite ordonner cet ensemble pour obtenir l'ensemble ordonné
    des complétions :math:`S=(s_i)` qu'on considère comme une permutation
    :math:`\sigma` de l'ensemble de départ : :math:`S(\sigma) = (s_i) = (c_{\sigma(j)})`.
    Ce système de complétion est destiné à un des utilisateurs qui forment des recherches ou requêtes
    :math:`Q=(q_i, w_i)_{1 \infegal i \infegal N_Q}`.
    :math:`q_i` est la requête, :math:`w_i` est la fréquence associée
    à cette requête. On définit l'effort demandé aux utilisateurs
    par ce système de complétion :

    .. math::

        E(C, Q, \sigma) = \sum_{i=1}^{N_Q} w_i M'(q_i, S(\sigma))

    Déterminer le meilleur système de complétion revient à trouver
    la permutation :math:`\sigma` qui minimise :math:`E(C, Q, \sigma)`.

La métrique :math:`M'` peut être remplacée par :math:`M"`. La différence
peut paraître insignifiante mais elle ne l'est pas tant que ça. Le système
de complétion peut se concevoir comme une *compression* :
le système de complétion permet de coder l'ensemble des recherches
d'un utilisateur :math:`Q` avec moins de caractères que celui-ci
en a besoin pour les taper. On ajoute les caractères :math:`rightarrow`
et :math:`\downarrow` aux lettres de l'alphabet et cela permet de
coder plus rapidement une requêtes. La quantité suivante peut être
considérée comme le taux de compression :

.. math::

    t(C, Q, \sigma) = \frac{ E(C, Q, \sigma) } { \sum_{i=1}^{N_Q} w_i l(q_i) }

L'idée derrière cette métaphore est le fait d'avoir une idée de la borne inférieure
pour ce taux de compression. On s'inspirer de la
`complexité de Lempel-Ziv <https://fr.wikipedia.org/wiki/Complexit%C3%A9_de_Lempel-Ziv>`_
(`calculating Lempel-Ziv (LZ) complexity (aka sequence complexity) of a binary string <http://stackoverflow.com/questions/4946695/calculating-lempel-ziv-lz-complexity-aka-sequence-complexity-of-a-binary-str>`_)
ou du `codage de Huffman <https://fr.wikipedia.org/wiki/Codage_de_Huffman>`_.
:math:`M'` permet une compression avec perte et :math:`M"` sans perte.
Le calcul de :math:`M'` autorise deux *jumps* de suite :

.. math::

    abb (\downarrow \downarrow \downarrow \downarrow \downarrow)

Mais les deux dernières touches :math:`\downarrow` peuvent s'appliquer
au premier préfixe ou aux suggestions montrées par la complétion
obtenue après trois :math:`\downarrow`.

.. math::

    abb (\downarrow \downarrow \downarrow) (\downarrow \downarrow)

La métrique :math:`M"` interdit ce cas.

Enoncé 2
++++++++

.. mathdef::
    :title: Optimiser un système de complétion filtré
    :lid: optim-nlp-comp2
    :tag: Problème

    On suppose que l'ensemble des complétions :math:`C=\acc{c_j}` est connu.
    On souhaite ordonner cet ensemble pour obtenir l'ensemble ordonné
    des complétions :math:`S=(s_i)` qu'on considère comme une permutation
    :math:`\sigma` de l'ensemble de départ : :math:`S(\sigma) = (s_i) = (c_{\sigma(j)})`.
    On utilise aussi une fonction :math:`f` qui filtre les suggestions montrées
    à l'utilisateur, elle ne change pas l'ordre mais peut cacher certaines suggestions
    si elles ne sont pas pertinentes.
    Ce système de complétion est destiné à un des utilisateurs qui forment des recherches ou requêtes
    :math:`Q=(q_i, w_i)_{1 \infegal i \infegal N_Q}`.
    :math:`q_i` est la requête, :math:`w_i` est la fréquence associée
    à cette requête. On définit l'effort demandé aux utilisateurs
    par ce système de complétion :

    .. math::

        E(C, Q, \sigma, f) = \sum_{i=1}^{N_Q} w_i M'(q_i, S(\sigma), f)

    Déterminer le meilleur système de complétion revient à trouver
    la permutation :math:`\sigma` qui minimise :math:`E(C, Q, \sigma, f)`.

Comme suggéré au paragraphe :ref:`l-nlp-comp-montre`, le filtre :math:`f`
peut rejetter une suggestion si elle est montrée à une position
qui ne permet aucun gain à l'utilisateur, c'est-à-dire que la différence
des longueurs complétion - préfixe est plus petite que la position où elle est montrée.

Une idée
++++++++

On aimerait bien pouvoir trouver l'ordre optimal par morceau,
supposer que l'ordre optimal pour l'ensemble des complétions
correspond à l'ordre des complétions sur un sous-ensemble
partageant le même préfixe.

.. mathdef::
    :title: M' et sous-ensemble
    :tag: Lemme
    :lid: lemme-nlp-m-sous-ens

    On suppose que la complétion :math:`q` est préfixe
    pour la requête :math:`q'` et
    :math:`\sigma(q) < \sigma(q')` ce qui signifie
    que la complétion :math:`q` est toujours affichée
    avant la complétion :math:`q'` si elles apparaissent ensemble.
    Alors :math:`M'(q, S) < M'(q', S)`.
    Plus spécifiquement, si on considère l'ensemble
    :math:`S'(q) = \acc{ s-q \in S | q \prec s }`
    (:math:`s-q` est la complétion :math:`s`
    sans son préfixe :math:`q`).

    .. math::

        M'(q', S) = M'(q'-q, S') + M'(q, S)

On sait déjà, par construction que
:math:`M'(q', S) \infegal M'(q'-q, S') + M'(q, S)`.
Par l'absurde, on suppose que :math:`M'(q', S) < M'(q'-q, S') + M'(q, S)`,
comme la requête :math:`q-q'` est toujours affichée avant la requête
:math:`q'`, cela voudrait dire qu'on aurait trouvé une façon plus optimale
d'écrire la requête :math:`q-q'` avec le système :math:`S` ce qui
impossible d'après la définition de la métrique :math:`M'`.
Cette propriété n'aide pas forcmément à trouver un algorithme
pour optimiser l'ordre des complétions dans la mesure où la
propriété suppose qu'une complétion soit affiché avant toutes
celles dont elle est le préfixe. La propriété suivante est évidemment vraie
pour le cas particulier qu'on vient de mentionner. Si elle est vraie, cela devrait
permettre de procéder par sous-ensemble pour trouver l'ordre optimal.

.. mathdef::
    :title: M', ordre et sous-ensemble
    :tag: Théorème
    :lid: lemme-nlp-m-sous-ens-ordre

    Soit :math:`q` une requête de l'ensemble de complétion :math:`S`
    ordonnées selon :math:`sigma`.
    Si cet ordre vérifie :

    .. math::
        :label: best-order-lemme-completion

        \forall k, \; \sigma(q[1..k]) \infegal \sigma(q[1..k+1])

    On note l'ensemble :math:`S'(q[1..k]) = \acc{ q[k+1..len(q)] \in S }` :

    alors :

    .. math::

        \forall k, \; M'(q[1..k], S) = M'(q[k+1..l(q)], S'(q[1..k]) + M'(q[1..k], S)

Ceci découle de l'application du lemme précédent.
Ce théorème permet presque de déterminer le meilleur ordre `\sigma` parmi ceux qui
vérifie la contrainte :eq:`best-order-lemme-completion`, à savoir
une requête courte est toujours affichée avant celles qui la complètent.
On procède par récurrence, on suppose connu les ordres :math:`\sigma(q)`
pour l'ensemble des complétions qui commencent par le préfixe :math:`p = q[1..k]`,
:math:`S'(q[1..k]) = \acc{ q | q[1..k] = p, q \in S }`. Pour :math:`i =k-1`,
le meilleur ordre : :math:`\sigma` revient à fusionner les listes ordonnées
obtenues pour chaque préfixe de longueur :math:`k`.
Il faut démontrer la possibilité de traiter les complétions par ordre croissant.
