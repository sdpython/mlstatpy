

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


.. contents::
    :local:


Formalisation
=============

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
    :title: Minimum keystroke
    :tag: Définition
    :lid: def-mks
    
    On définit la façon optimale de saisir une requête sachant un système de complétion
    :math:`S` comme étant le minimum obtenu :
    
    .. math::
        
        M(q,S) = \min_{0 \infegal k \infegal l(q)}  k + M(q, k, S)
        
    La quantité :math:`M(q, k, S)` représente le nombre de touche vers le bas qu'il faut taper pour
    obtenir la chaîne :math:`q` avec le système de complétion :math:`S` et les :math:`k`
    premières lettres de :math:`q`.


De façon évidente, :math:`S(q, l(q))=0`.
Certains systèmes proposent des requêtes avant de saisir quoique ce soit,
c'est pourquoi on inclut la valeur :math:`S(q, 0)` qui représente ce cas.
Construire un système de complétion revient à minimiser la quantité :

.. math::

    M(S) = \sum_{i=1}^N M(q_i,S) w_i


Ensemble des requêtes
+++++++++++++++++++++

Il n'y a pas de restriction sur la fonction :math:`M(q, k, S)` mais on se limitera
dans un premier temps à une fonction simple. On suppose que le système d'autocomplétion
dispose d'un ensemble de requêtes ordonnées :math:`S = (s_i)` et la fonction :

.. math::

    M(q, k, S) = position(q, S(q_k))
    
Où :math:`S(q_k)` est le sous-ensemble ordonné de :math:`S` des requêtes
qui commence par les :math:`k` premières lettres de :math:`q` et de longueur supérieure strictement à :math:`k`.
:math:`position(q, S(q_k))` est la position de :math:`q` dans cet ensemble ordonné
ou :math:`\infty` si elle n'y est pas.

.. math::

    M(q, k, S) = \min\acc{ i | s_i \succ q[1..k]  }
    
Trouver le meilleur système de complétion :math:`S` revient à trouver la meilleure
fonction :math:`M(q, k, S)` et dans le cas restreint l'ordre sur :math:`S` qui minimise
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
    
Dans tous les cas, :math:`M(q, k, S) = l(q) - k`. Cela veut dire
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
abcd   1         1      1 = :math:`M(q, 0, S)`
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

Le premier exemple offre aussi un contre exemple.
Dans cet exemple, l'ensemble :math:`Q=(q_i)` des
requêtes utilisateurs et l'ensemble :math:`S=(s_i)`
des requêtes complétées est le même.
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
Dans quel ordre faut-il ranger les requêtes complétées pour économiser le
plus de caractères. On aurait tendance à dire la plus longue d'abord
ce qu'on peut vérifier dans le notebook qui suit :

.. toctree::
    :maxdepth: 1
    
    notebooks/completion_trie


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



Nouvelle métrique
=================


Notion de trie
==============


Vocabulaire
===========
