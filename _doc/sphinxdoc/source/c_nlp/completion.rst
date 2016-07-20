

.. _l-completion0:

==========
Complétion
==========

.. index:: complétion, whoosh

La `complétion <https://fr.wikipedia.org/wiki/Compl%C3%A8tement>`_ est un méchanisme
qui permet à un utilisateur de saisir les mots de sa recherche avec moins
de caractères qu'elle n'en contient. L'utilisateur saisit plus rapidement.

.. image:: completion_img/wiki.png


Si ces outils sont appréciables du point de vue utilisateurs,
ils le sont tout autant côté site web en réduisant la variabilité dans
le texte saisie, en particulier les fautes d'orthographes. L'utilisateur
a besoin de moins de requêtes pour trouver son produits et cela diminue 
d'autant la charge du serveur qui lui fournit ses résultats.

Ce chapitre aborde différentes problématiques liées à ce genre de systèmes
qui sont présents partout sur Internet, 
moteurs de recherches, sites de ventes en ligne, journaux...
Il existe de nombreuses librairies qui les implémentent.
La plus connue en Python est `whoosh <https://whoosh.readthedocs.io/en/latest/>`_.

Quelques éléments de codes sont disponibles dans le module
:mod:`completion <mlstatpy.nlp.completion>` et le notebook 
:ref:`completiontrierst`.


.. contents::
    :local:


Formalisation
=============

.. _l-completion-optim:

Problème d'optimisation
+++++++++++++++++++++++

Je me réfère pour cela à l'article [Sevenster2013]_ qui introduit différentes façons de construire
un système d'autocompétion et qui les compare à l'usage. Et s'il existe plusieurs façons de faire, il 
faut d'abord mettre au point une façon de les comparer.
Je me place dans le cadre d'un moteur de recherche car c'est l'usage principal,
que celui-ci soit un moteur de recherche ou une recherche incluse sur un site de vente.
A la fin de la journée, on sait quelles sont les requêtes saisies par les utilisateurs
et le nombre de fois qu'elles ont été saisies : :math:`(q_i, w_i)` pour
:math:`i \in [[1, N]]`. 

.. index:: caractère saisi, keystroke

Sans système de complétion, les utilisateurs saisissent donc :math:`K=\sum_{i=1}^N l(q_i) w_i`
où :math:`l(q_i)` est la longueur de la requête :math:`q_i`. Avec le système de complétion,
les utilisateurs saisissent moins de caractères, c'est ce chiffre là qu'on cherche à minimiser.
L'unité est le charactère saisi ou *keystroke* en anglais.

Même avec le même système de complétion, 
il n'est pas dit que tous les utilisateurs saisissent la même requête de la même
façon. Pour simplifier, on va supposer que si malgré tout et ne considérer que la façon
minimale de saisir une requête.

.. image:: completion_img/comp.png

L'exemple précédent illustrent deux façons de saisir le terme *autocomplétion* (sur Wikipédia),
*autocom* + 4 touches vers le bas ou *autocomp* + 1 touche vers le bas, soit 7+4=11 touches 
dans le premier cas ou 8+1=9 touches dans le second cas. 

.. mathdef::
    :title: Minimum Keystroke
    :tag: Définition
    :lid: def-mks
    
    On définit la façon optimale de saisir une requête sachant un système de complétion
    :math:`S` comme étant le minimum obtenu :
    
    .. math::
        :label: completion-metric1
        
        M(q,S) = \min_{0 \infegal k \infegal l(q)}  k + K(q, k, S)
        
    La quantité :math:`K(q, k, S)` représente le nombre de touche vers le bas qu'il faut taper pour
    obtenir la chaîne :math:`q` avec le système de complétion :math:`S` et les :math:`k`
    premières lettres de :math:`q`.


De façon évidente, :math:`K(q, l(q), S)=0` et :math:`M(q,S) \infegal l(q)`.
Certains systèmes proposent des requêtes avant de saisir quoique ce soit,
c'est pourquoi on inclut la valeur :math:`M(q, 0)` qui représente ce cas.
Construire un système de complétion revient à minimiser la quantité :

.. math::

    M(S) = \sum_{i=1}^N M(q_i,S) w_i


Ensemble des requêtes
+++++++++++++++++++++

Il n'y a pas de restriction sur la fonction :math:`K(q, k, S)` mais on se limitera
dans un premier temps à une fonction simple. On suppose que le système d'autocomplétion
dispose d'un ensemble de requêtes ordonnées :math:`S = (s_i)` et la fonction :

.. math::

    K(q, k, S) = position(q, S(q_k))
    
Où :math:`S(q_k)` est le sous-ensemble ordonné de :math:`S` des requêtes
qui commence par les :math:`k` premières lettres de :math:`q` et de longueur supérieure strictement à :math:`k`.
:math:`position(q, S(q_k))` est la position de :math:`q` dans cet ensemble ordonné
ou :math:`\infty` si elle n'y est pas. Cette position est strictement positive
:math:`K(q, k, S) \supegal 1` sauf si :math`k=l(q)` auquel cas, elle est nulle. 
Cela signifie que l'utilisateur doit descendre d'au moins un cran
pour sélectionner une suggestion.

.. math::

    K(q, k, S) = \min\acc{ i | s_i \succ q[1..k], s_i \in S }
    
Trouver le meilleur système de complétion :math:`S` revient à trouver la meilleure
fonction :math:`K(q, k, S)` et dans le cas restreint l'ordre sur :math:`S` qui minimise
cette fonction. Le plus souvent, on se contente de trier les requêtes par ordre
décroissant de popularité. On considérera par la suite qu'on est dans ce cas.

Gain
++++

On définit le gain en keystroke comme étant le nombre de caractères saisis en moins :

.. math::

    G(q, S) = l(s) - M(q,S)
    
Minimier :math:`M(S)` ou maximiser :math:`G(S) = \sum_{i=1}^N G(q_i, S) w_i` 
revient au même.

.. math::

    G(S) = \sum_{i=1}^N w_i (l(s) - M(q,S)) = \sum_{i=1}^N w_i l(s) - \sum_{i=1}^N w_i  M(q,S))  = K - M(S)

Où :math:`K=\sum_{i=1}^N l(q_i) w_i` l'ensemble des caractères tapés par les utilisateurs.
:math:`\frac{G(S)}{K}` est en quelque sorte le ratio de caractères économisés
par le système de complétion.



.. [Sevenster2013] Algorithmic and user study of an autocompletion algorithm on a large
    medical vocabulary (2013), 
    Merlijn Sevenster, Rob van Ommering, Yuechen Qian
    *Journal of Biomedical Informatics* 45, pages 107-119


Fausses idées reçues
====================

Il faut trier les requêtes par fréquence décroissante
+++++++++++++++++++++++++++++++++++++++++++++++++++++

En pratique, cela marche plutôt bien. En théorie, cette assertion est fausse.
Prenons les quatre requêtes suivantes :

====== ========= ======
q      fréquence ordre
====== ========= ======
a      4         1
ab     3         2
abc    2         3
abcd   1         4
====== ========= ======

Dans cet exemple, si l'utilisateur tape ``ab``, il verra les requêtes :

::

    abc
    abcd
    
Dans tous les cas, :math:`K(q, k, S) = l(q) - k`. Cela veut dire
que l'utilisateur ne gagnera rien. En revanche, avec l'ordre suivant :

====== ======
q      ordre
====== ======
a      4
ab     2
abc    3
abcd   1
====== ======

Si l'utilisateur tape ``ab``, il verra les requêtes :

::

    abcd
    abc

Le nombre de caractères économisés sera :

====== ========= ====== ====================== 
q      fréquence ordre  :math:`M(q, S)`
====== ========= ====== ====================== 
a      4         4      1
ab     3         2      2
abc    2         3      3
abcd   1         1      1 = :math:`K(q, 0, S)`
====== ========= ====== ====================== 

D'où un gain total de :math:`G(S)=3`.


Il faut placer les requêtes courtes avant
+++++++++++++++++++++++++++++++++++++++++

Le cas précédent est déjà un contre exemple. 
Mais d'un point de vue utilisateur, il n'est pas facile de lire
des requêtes de longueurs différentes. Cela veut peut-être dire aussi
que la métrique considérée pour choisir le meilleur système de complétion
est faux. Cela sera discuté à la prochaine section.

Il faut compléter toutes les requêtes
+++++++++++++++++++++++++++++++++++++

.. index:: requête complète

Le premier exemple offre aussi un contre exemple.
Dans cet exemple, l'ensemble :math:`Q=(q_i)` des
requêtes utilisateurs et l'ensemble :math:`S=(s_i)`
des **requêtes complètes** est le même.
Il suffit de la modifier un peu. On enlève 
la requête *ab* de :math:`S`.


====== ========= ============== ================ 
q      fréquence ordre          :math:`M(q, S)`
====== ========= ============== ================ 
a      4         1              1
ab     3         :math:`\infty` 2
abc    2         2              2
abcd   1         3              3
====== ========= ============== ================ 

D'où un gain total de :math:`G(S)=2`. En conclusion,
si j'enlève une petite requête pour laquelle le gain est nul,
il est possible que le gain pour les suivantes soit positif.
On en retient qu'il ne faut pas montrer trop de requêtes 
qui se distinguent d'un caractère.


Et si le poids de chaque requête est uniforme
+++++++++++++++++++++++++++++++++++++++++++++

On suppose que les requêtes ont toutes le même poids :math:`w_i=1`.
Dans quel ordre faut-il ranger les requêtes complètes pour économiser le
plus de caractères. On aurait tendance à dire la plus longue d'abord
ce qu'on peut vérifier dans le notebook :ref:`completiontrierst`.


====== ========= ============== ================
q      fréquence ordre          :math:`M(q, S)`
====== ========= ============== ================
a      1         4              1
ab     1         3              2
abc    1         2              2
abcd   1         1              1
====== ========= ============== ================

Ajouter deux autres requêtes disjointes *edf*, *edfh*.
Le gain maximum est 6 et il y a plusieurs ordres :

::

    'edf', 'edfh', 'abc', 'abcd', 'a', 'ab'
    'abcd', 'abc', 'edfh', 'edf', 'ab', 'a'
    ...
    
On a presque l'impression qu'on peut traiter chaque bloc
séparément *a, ab, abc, abcd* d'un côté et *edf, edfh* de l'autre.
A l'intérieur des blocs, les règles seront les mêmes.

.. image:: completion_img/trieex.png

En résumé, si on connaît le meilleur ordre pour toutes les mots sur les noeuds 
temrinaux dans les bulles rouges, dans la bulle verte, le meilleur ordre
sera une fusion des deux listes ordonnées.

Quelques essais sur le notebook ont tendance à montrer que l'ordre
a peu d'impact sur le résultat final lorsque les requêtes ont le même poids.
Avec quatre mots, la somme des gains est identique quelque soit l'ordre.

::

    p=poids g=gain

    20.0 - actuellement p=1.0 g=11.0 | acte p=1.0 g=2.0 | actes p=1.0 g=2.0 | actualité p=1.0 g=5.0
    20.0 - acte p=1.0 g=3.0 | actuellement p=1.0 g=10.0 | actualité p=1.0 g=6.0 | actes p=1.0 g=1.0
    20.0 - acte p=1.0 g=3.0 | actes p=1.0 g=3.0 | actualité p=1.0 g=6.0 | actuellement p=1.0 g=8.0

Mais si on change le poids de l'une d'elles, elle se retrouve en première position.

::

    19.2 - actes p=2.0 g=4.0 | actuellement p=1.0 g=10.0 | acte p=1.0 g=1.0 | actualité p=1.0 g=5.0
    19.2 - actes p=2.0 g=4.0 | actuellement p=1.0 g=10.0 | actualité p=1.0 g=6.0 | acte p=1.0 g=0.0


Intuitions
++++++++++

#. La métrique actuelle n'est pas la meilleure.
#. Si les mots n'ont pas de long préfixes en commun, il vaut mieux
   placer le mot le plus fréquent en première position.
   Pour les mots de fréquence identique, l'ordre a peu d'importance.
#. S'il existe une séquence de mots emboîtés, les gains sont minimes
   à moins d'enlever des mots ou de poser les grandes requêtes d'abord.

Les intuitions 2 et 3 seront sans doute remise en question en considérant 
une nouvelle métrique.


Nouvelle métrique
=================

Intuition
+++++++++

On considère l'ensemble des requêtes complètes
:math:`S` composé de deux mots *actuellement*, *actualité*.
Le gain moyen par mots est de 9 caractères économisés.
Si on ajoute le grand préfixe commun à la liste *actu*,
ce gain moyen tombe à 6.33 (voir :ref:`completiontrierst`) quelque
soit l'ordre choisi pour les requêtes. Toutefois, si on ne prend pas 
en compte le gain sur le mot *actu* car ce n'est pas un mot 
correct mais plus un mot qui aide la lecture de la liste, ce gain
moyen tombe à 8 seulement. En conclusion, si l'utilisateur 
tape la lettre **a** et qu'on lui montre ceci :

::

    actu
    actualité
    actuellement

Au lieu de :

::

    actualité
    actuellement
    
Il doit taper en moyenne un caractère de plus pour obtenir le mot qu'il cherche.
Et la métrique ne montre pas réellement de préférence pour l'ordre d'affichage
des requêtes. Pourtant, l'utilisateur pourrait très bien utiliser la 
séquence de touches suivantes : 

=========== =================
touche      mot composé
=========== =================
a           a
bas         actu (suggestion)
e           actue
bas         actuellement
=========== =================

Dans cet exemple aussi petit, on ne gagnerait pas grand-chose
mais cela vaut le coup d'étudier cette piste pour des vocabulaires plus
grand : se servir des préfixes commun comme tremplin pour les mots
plus grand. L'effect position perdrait un peu de son influence.

Formalisation
+++++++++++++

On reprend la première métrique :eq:`completion-metric1` :

.. math::
    :nowrap:

    \begin{eqnarray*}
    M(q, k, S) &=& \min\acc{ i | s_i \succ q[1..k], s_i \in S } \\
    M(q, S) &=& \min_{0 \infegal k \infegal l(q)}  k + K(q, k, S)
    \end{eqnarray*}
    
:math:`M(q, k, S)` définit la position de la requête :math:`q`
dans la liste affichée pour le préfixe composé des :math:`k` premières lettres
de :math:`q`. On va juste changer :math:`k` dans la seconde en ligne.


.. mathdef::
    :title: Dynamic Minimum Keystroke
    :tag: Définition
    :lid: def-mks2
    
    On définit la façon optimale de saisir une requête sachant un système de complétion
    :math:`S` comme étant le minimum obtenu :
    
    .. math::
        :label: completion-metric2
        :nowrap:
        
        \begin{eqnarray*}
        K(q, k, S) &=& \min\acc{ i | s_i \succ q[1..k], s_i \in S } \\
        M'(q, S) &=& \min_{0 \infegal k \infegal l(q)} \acc{ M'(q[1..k], S) + K(q, k, S) | q[1..k] \in S }
        \end{eqnarray*}

De manière évidente, :math:`M'(q, S) \infegal M(q, S)`.
Il reste à démontrer que cette métrique et bien définie puisqu'elle
fait partie de sa définition. La condition :math:`q[1..k] \in S` impose que
le préfixe composé des *k* premières lettres :math:`q[1..k]` fasse partie 
des requêtes complètes :math:`S`. Dans le cas contraire, elle n'est pas
affichée et l'utilisateur ne pourra pas s'en servir comme tremplin.

Si on définit la quantité :math:`M_0(q, S) = M(q, S)` et par récurrence :

.. math::

    M_{t+1}(q, S) = \min_{0 \infegal k \infegal l(q)} \acc{ M_t(q[1..k], S) + K(q, k, S)  | q[1..k] \in S }
    
La suite :math:`(M_t(q, S))_t` est décroissante et positive. Elle converge nécessaire
vers la valeur cherchée :math:`M'(q, S)`. Cela donne aussi une idée de la façon de le calculer.
Contrairement à la première métrique, le calcul dépend du résultat pour 
tous les préfixes d'une requête. Il ne peut plus être calculé indépendemment.





Notion de trie
==============

Une implémentation des tries est décrites dans deux notebooks :
`Arbre et Trie <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/notebooks/_gs1a_A_arbre_trie.html>`_.
Les résultats de ce chapitre ont été produits avec le module :mod:`completion <mlstatpy.nlp.completion>`
et le notebook :ref:`completiontrierst`.

Remarques pour optimiser les calculs
++++++++++++++++++++++++++++++++++++

**K(q, k, S)**

On reprend la première métrique :eq:`completion-metric1` :

.. math::
    :nowrap:

    \begin{eqnarray*}
    K(q, k, S) &=& \min\acc{ i | s_i \succ q[1..k], s_i \in S } \\
    M(q, S) &=& \min_{0 \infegal k \infegal l(q)}  k + K(q, k, S)
    \end{eqnarray*}

Etant donné que le nombre minimum de caractères pour obtenir une requête dans le trie
ne peut pas être supérieur à la longueur, si :math:`K(q, k, S) > l(q) - k`, on sait déjà que
que le préfixe :math:`q[1..k]` ne sera pas le minimum.

**suggestions**

On considère les requêtes complètes suivante :

::

    actu
    actualité
    actualités
    actuel
    actuellement
    
Pour le préfixe *actue*, on suggère *actuel* at *actuellement*.
Pour le préfixe *actua*, on suggère *actualité* at *actualités*.
Pour le préfixe *actu*, on suggère la concaténation de ces deux listes.
Par conséquent, pour construire les listes de suggestions associées à chaque préfixe,
il paraît de partir des feuilles de l'arbre puis de fusionner les listes
de suggestions jusqu'au noeud racine.

**utilisation ou recherche**

C'est différent de construire toutes le suggestions pour un préfixe plutôt 
que toutes les suggestions pour tous les préfixes. Le premier cas correspond
à un utilisateur qui cherche quelque chose. Il faut être rapide quitte à retourner un 
résultat tronqué.

Le second cas correspond à objectif de recherche des d'optimisation.
Les enjeux sont plus de réussir à calculer toutes les suggestions
en un temps raisonnable et avec une utilisation mémoire raisonnable également.

**mémoire**

D'après la remarque précédente, il n'est pas utile de conserver pour un préfixe donné
l'intégralité des requêtes complètes qui commence par ce préfixe. Dans le pire des cas,
cette liste a besoin de contenir autant de suggestions que le nombre de caractères de la
plus longue requêtes.


Vocabulaire
===========

Synonymes
+++++++++

On utilise dabord les préfixes pour chercher les mots dans un trie 
mails il est tout à fait possible de considérer des synonymes.
Avec les préfixes, un noeud a au plus 27 (26 lettres + espaces) 
caractères suivant possibles. Si le préfixe a des synonymes,
rien n'empêche de relier ce noeud avec les successeurs de ses
synonymes.

Source
++++++

Dans le cas d'un moteur de recherche, le trie ou l'ensemble :math:`S` des requêtes complètes
est construit à partir des requêtes des utilisateurs. Lorsque le système
de complétion est mise en place, la distribution des requêtes changent. Les requêtes
les plus utilisées vont être encore plus utilisées car les utilisateurs vont moins
s'égarer en chemin comme s'égarer vers une faute d'orthographe.

