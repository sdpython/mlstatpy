
Digressions
===========

.. contents::
    :local:

Synonymes, Contexte
+++++++++++++++++++

On utilise dabord les préfixes pour chercher les mots dans un trie
mails il est tout à fait possible de considérer des synonymes.
Avec les préfixes, un noeud a au plus 27 (26 lettres + espaces)
caractères suivant possibles. Si le préfixe a des synonymes,
rien n'empêche de relier ce noeud avec les successeurs de ses
synonymes.
A ce sujet, voir `Context-Sensitive Query Auto-Completion <http://technion.ac.il/~nkraus/papers/fr332-bar-yossef.pdf>`_,
de Ziv Bar-Yossef et Naama Kraus.

Source
++++++

Dans le cas d'un moteur de recherche, le trie ou l'ensemble :math:`S` des requêtes complètes
est construit à partir des requêtes des utilisateurs. Lorsque le système
de complétion est mise en place, la distribution des requêtes changent. Les requêtes
les plus utilisées vont être encore plus utilisées car les utilisateurs vont moins
s'égarer en chemin comme s'égarer vers une faute d'orthographe.
Comment corriger la distribution des requêtes malgré l'intervention
du système de complétion ? Cela pourrait faire l'objet d'un sujet de recherche.

Fonction de gain
++++++++++++++++

Jusqu'à présent, on a considéré uniquement le nombre de caractères économisés pour
déterminer le meilleur ordre. Rien n'empêche d'ajouter une coût supplémenaires lié
à l'ordre des complétions. Une requête est pénalisée si les complétions
associées sont loin de l'ordre alphabétique. On peut pénaliser un ordre éloigné
à chaque caractère ajouté.

Minuscules, majuscules
++++++++++++++++++++++

C'est bien connu, on fait peu de ces des accents sur internet.
De fait, même si l'accent apparaît à l'écran, le système de complétion
verra peut de différences entre le ``e`` et ``é``.
Sur Wikpédia, les homonymes sont distingués par un sous-titre
entre parenthèse l'année pour un événement sportif régulier.
On peut imaginer que plusieurs séquences de caractères aboutissent
à la même entrée.

Suppression de caractères
+++++++++++++++++++++++++

Nous pourrions considérer le fait de pouvoir supprimer des caractères
afin de trouver le chemmin le plus court pour obtenir une requête.

Coût d'un caractère
+++++++++++++++++++

Jusqu'à présent, la pression d'une touche a le même coût quelque soit
la source, un caractère, une touche vers le bas. Pourtant, plus il y a
de lettres dans l'alphabet, plus le système de complétion sera performant
à supposer que les mots soient plus ou moins équirépartis selon les
caractères (la probabilité du prochain caractère est uniforme).
On peut concevoir que chercher une touche lorsque l'alphabet est grand peut prendre
un certain temps. Le cas du chinois est intéressant car la
`saisie des caractères <https://fr.wikipedia.org/wiki/Saisie_du_chinois_sur_ordinateur>`_
peut prendre plusieurs touches. Faut-il considérer un caractère chinois comme unité
de décomposition d'un mot où la séquence des touches qui le construisent ?
Dans le premier cas, il faudrait sans doute pénaliser la saisie d'un caractère
en fonction du nombre de touches nécessaires pour le former par rapport
à la sélection d'une complétion.

Complétion partielle
++++++++++++++++++++

On rappelle la métrique :eq:`completion-metric2` (voir aussi :eq:`nlp-comp-k`).

.. math::
    :nowrap:

    \begin{eqnarray*}
    M'(q, S) &=& \min_{0 \infegal k \infegal l(q)} \acc{ M'(q[1..k], S) + K(q, k, S) }
    \end{eqnarray*}

Si on note :math:`L(p, S)` l'ensemble des complétions
pour le préfixe :math:`p`.
Que dire de la définition suivante ?

.. math::
    :nowrap:

    \begin{eqnarray*}
    M'_p(q, S) &=& \min_{0 \infegal k \infegal l(q)} \acc{ \begin{array}{l}
                            \indicatrice{ L(q[1..k], S) \neq \emptyset} \cro{M'_p(q[1..k], S) +  K(q, k, S)} + \\
                            \;\;\;\;\indicatrice{L(q[1..k], S) = \emptyset} \cro { \min_j M'_p(q[1..j], S) + M'_p(q[j+1..], S)  }
                            \end{array} }
    \end{eqnarray*}

Cela revient à considérer que si le système de complétion ne propose aucune complétion
avec le préfixe en cours, on propose des complétions avec un préfixe
qui ne tient compte que des dernières lettres.

Utilisateurs
++++++++++++

La modélisation mathématique aboutit à l'optimisation
d'une métrique qui ne coïncide pas forcément à la logique
de l'utilisateur. Le fait de montrer une suggestion dans la liste
donne en quelques sortes un signal qui indique que cette requête
a une bonne d'aboutir. C'est en particulier vrai si elle est identique
à celle que l'utilisateur est en train de saisir quand même même elle
n'aurait aucun sens. La présence d'une suggestion conforte l'utilisateur
dans le chemin qu'il choisit.

Un autre aspect est le fait qu'une suggestion, la première par exemple,
doit rester à l'écran si celle-ci inclut le préfixe saisit
par l'utilisateur. Supposons que ce dernier saisisse *fac*,
le moteur de suggestions lui montrera en premier *facebook*.
Il faudra que *facebook* demeure en première position jusqu'à ce
l'utilisateur ait soit cliqué dessus, soit saisi quelque chose
de plus long. Enlever la suggestion *facebook* alors que le préfixe
en cours est *faceboo* perturbe.

Revenir en arrière
++++++++++++++++++

On pourrait également imaginer une métrique qui calcule
le meilleur chemin en donnant la possibilité de revenir en
arrière. Par exemple, pour saisir *éléphant*, un utilisateur
pourrait utiliser la suggestion *éléphants* puis enlever le dernier *s*.
On note la métrique :math:`M'_b`.

.. mathdef::
    :title: Dynamic Minimum Keystroke arrière
    :tag: Définition
    :lid: def-mks2

    On définit la façon optimale de saisir une requête
    sachant un système de complétion
    :math:`S` comme étant le minimum obtenu :

    .. math::
        :label: completion-metric2
        :nowrap:

        \begin{eqnarray*}
        M'_b(q, S) &=& \min\acc{\begin{array}{l}
                \min_{0 \infegal k < l(q)} \acc{ M'_b(q[1..k], S) +
                    \min( K(q, k, S), l(q) - k) } \\
                \min_{s \succ q} \acc{ M'_b(s, S) + l(s) - l(q) }
                \end{array} }
        \end{eqnarray*}

Le second terme de la métrique nécessite de considérer
toutes les suggestions dans *S* qui commencent par le préfixe *q*.
Le calcul de la métrique peut être fait en adaptant l'algorithme
reprend les idées du paragraphe :ref:`mp-completion-metrique-formalisation`
mais il faudra parcourir les suggestions dans l'ordre alphabétique
puis mettre à jour la métrique en les parcourant dans l'autre sens.
On recommence jusqu'à convergence.
