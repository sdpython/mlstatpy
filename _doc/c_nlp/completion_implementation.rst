
Implémentation
==============

.. contents::
    :local:

.. _trie: https://fr.wikipedia.org/wiki/Trie_(informatique)

J'allais vous raconter en détail ce qu'est un trie_ et le paragraphe suivant
vous en dira sans doute un peu plus à ce sujet. Le trie_ est le moyen
le plus efficace de trouver un mot aléatoire ou un préfixe aléatoire dans une liste.
Mais il y a mieux et plus simple dans notre cas où il faut trouver
une longue liste de mots connue à l'avance - donc pas aléatoire -.
Et puis, c'était sous mes yeux. Il y a plus simple et aussi efficace quand
les listes des mots et des complétions sont connues à l'avance.

Notion de trie
++++++++++++++

Une implémentation des tries est décrites dans deux notebooks :
`Arbre et Trie <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/td1a_cenonce_session8.htmlhttp://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/_gs1a_A_arbre_trie.html>`_.
Les résultats de ce chapitre ont été produits avec le module :mod:`completion <mlstatpy.nlp.completion>`
et le notebook :ref:`/notebooks/nlp/completion_trie.ipynb`. Le notebook
:ref:`/notebooks/nlp/completion_profiling.ipynb` montre les résultats du profiling.
L'implémentation Python est très gourmande en mémoire et elle serait
plus efficace en C++.

**utilisation ou recherche**

C'est différent de construire toutes les complétions pour un préfixe plutôt
que toutes les complétions pour tous les préfixes. Le premier cas correspond
à un utilisateur qui cherche quelque chose. Il faut être rapide quitte à retourner un
résultat tronqué.

Le second cas correspond à objectif de recherche des d'optimisation.
Les enjeux sont plus de réussir à calculer toutes les complétions
en un temps raisonnable et avec une utilisation mémoire raisonnable également.

**mémoire**

D'après la remarque précédente, il n'est pas utile de conserver pour un préfixe donné
l'intégralité des complétions qui commencent par ce préfixe. Dans le pire des cas,
cette liste a besoin de contenir autant de complétions que le nombre de caractères de la
plus longue complétioms.

Algorithme élégant
++++++++++++++++++

Il faut relire le premier problème d':ref:`optimisation <optim-nlp-comp>`
pour commencer à se poser la question : comment calculer la quantité
:math:`E(C, C, \sigma)` lorsque :math:`\sigma` correspond à l'ordre alphabétique ?
La réponse est simple : il suffit de parcourir les complétions une et une seule fois.
Supposons qu'au cours de ce parcours, on est à la complétion d'indice :math:`i`.
On conserve un compteur :math:`p(k, i)=K(c(i), k, C)` qui représente la position de la
complétion :math:`c(i)` dans la liste des complétions affichées par le système de complétion
pour le préfixe :math:`c(i)[[1..k]]`. Le coût de l'algorithme est en :math:`O(N\ln N + LN)` où
:math:`N` est le nombre de complétions et :math:`L` la longueur maximale d'une complétion.

Dans le cas où :math:`\sigma` est quelconque et :math:`C \neq Q`, on procède en deux étapes.
Dans un premier temps, on utilise une variante de l'algorithme précédent pour calculer
:math:`M'(q, C)` pour les requêtes :math:`q` dans l'ensemble des complétions.

Dans un second temps, on effectue une sorte de fusion entre les deux listes
triées alphabétiquement. Le coût de l'algorithme est en :math:`O(ILN + 2 N\ln N + M \ln M + max(N,M))`
où :math:`M` est le nombre de requêtes dans l'ensemble :math:`Q`. Cette partie repose sur le
:ref:`lemme <lemme-nlp-long-completion>` lié au calcul des métriques
pour les réquêtes hors de l'ensemble des complétions. :math:`I` est un nombre d'itération nécessaires
pour que les métriques :math:`M'` convergent pour l'ensemble des complétions. En pratique, c'est très petit.

L'algorithme est implémenté dans le module
:mod:`completion_simple <mlstatpy.nlp.completion_simple>` et plus particulièrement la fonction
:meth:`CompletionSystem.compute_metrics <mlstatpy.nlp.completion_simple.CompletionSystem.compute_metrics>`.
