
Nouvelle métrique
=================

Intuitions
++++++++++

#. La métrique actuelle n'est pas la meilleure.
#. Si les mots n'ont pas de long préfixes en commun, il vaut mieux
   placer le mot le plus fréquent en première position.
   Pour les mots de fréquence identique, l'ordre a peu d'importance.
#. S'il existe une séquence de mots emboîtés, les gains sont minimes
   à moins d'enlever des mots ou de poser les grandes complétions d'abord.

Les intuitions 2 et 3 seront sans doute remise en question en considérant 
une nouvelle métrique.
On considère l'ensemble des complétions
:math:`S` composé de deux mots *actuellement*, *actualité*.
Le gain moyen par mots est de 9 caractères économisés.
Si on ajoute le grand préfixe commun à la liste *actu*,
ce gain moyen tombe à 6.33 (voir :ref:`completiontrierst`) quelque
soit l'ordre choisi pour les complétions. Toutefois, si on ne prend pas 
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
des complétions. Pourtant, l'utilisateur pourrait très bien utiliser la 
séquence de touches suivantes : 

=========== =================
touche      mot composé
=========== =================
a           a
bas         actu (complétion)
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
    M(q, S) &=& \min_{0 \infegal k \infegal l(q)}  k + K(q, k, S)
    \end{eqnarray*}

La fonction :math:`K(q, k, S)` est définie par :eq:`nlp-comp-k`.


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
        M'(q, S) &=& \min_{0 \infegal k < l(q)} \acc{ M'(q[1..k], S) + 
                    \min( K(q, k, S), l(q) - k) }
        \end{eqnarray*}

On prend comme convention :math:`M'(\emptyset, S)=0`. Le calcul de la métrique
se construit comme une suite qui part des chaînes les plus courtes aux plus longues.
La métrique est donc bien définie.
Contrairement à la première métrique, le calcul dépend du résultat pour 
tous les préfixes d'une complétion. 

.. mathref::
    :title: métriques
    :tag: propriété


    :math:`\forall q, \; M'(q, S) \infegal M(q, S)`
    
Si :math:`q \notin S`, c'est évident puisque :math:`M'(q, S) \infegal M'(\emptyset, S) + l(q)`.
Si :math:`q \in S`, cela découle de la constation précédente puisque : 
:math:`M'(q, S) \infegal M'(q[[1..k]], S) + K(q, k, S) \infegal k + K(q, k, S)`.




Quelques résultats
++++++++++++++++++

On considère la liste des mots ``actuellement``, ``actualité``, ``actuel``.
On compare les ordres qui maximisent la première et la seconde
métriques ainsi que le gain obtenu. Première métrique ::

    7.0 - actuellement p=1.0 g=11.0 | actuel p=1.0 g=4.0 | actualité p=1.0 g=6.0
    7.0 - actuellement p=1.0 g=11.0 | actualité p=1.0 g=7.0 | actuel p=1.0 g=3.0
    7.0 - actuel p=1.0 g=5.0 | actuellement p=1.0 g=10.0 | actualité p=1.0 g=6.0

Seconde métrique ::

    7.333 - actuel p=1.0 g=5.0 | actualité p=1.0 g=7.0 | actuellement p=1.0 g=10.0
    7.0 - actuellement p=1.0 g=11.0 | actuel p=1.0 g=4.0 | actualité p=1.0 g=6.0

On note que la seconde métrique propose un meilleur gain, ce qui est attendu
mais aussi que le mot *actuel* sera placé devant le 
mot *actuellement*, plus long sans que cela souffre d'ambiguïté.

Définition avancée
++++++++++++++++++

Dans les faits, le :ref:`Dynamic Minimum Keystroke <completion-metric2>` sous-estime 
le nombre de caractères nécessaires. Lorsqu'on utilise un mot comme tremplin, on
peut aisément le compléter mais il faut presser une touche ou attendre un peu
pour voir les nouvelles complétions associées à la première complétion choisie et maintenant
considéré comme préfixe. C'est ce que prend en compte la définition suivante.

.. mathdef::
    :title: Dynamic Minimum Keystroke modifié
    :tag: Définition
    :lid: def-mks3
    
    On définit la façon optimale de saisir une requête sachant un système de complétion
    :math:`S` comme étant le minimum obtenu :
    
    .. math::
        :label: completion-metric3
        :nowrap:
        
        \begin{eqnarray*}
        M"(q, S) &=& \min \left\{ \begin{array}{l}
                        \min_{1 \infegal k \infegal l(q)} \acc{ M"(q[1..k-1], S) + 1 +\min( K(q, k, S), l(q) - k) } \\
                        \min_{0 \infegal k \infegal l(q)} \acc{ M"(q[1..k], S) + \delta + \min( K(q, k, S), l(q) - k) } 
                        \end{array} \right .
        \end{eqnarray*}

Si on prend comme exemple la requête *machine learning*, le premier cas correspond à la séquence :

* sélection de la complétion *machine*
* pression de la touche espace
* sélection de la complétion *machine learning*

Et le second cas à la séquence :

* sélection de la complétion *machine*
* pression de la touche droite pour afficher les nouvelles complétions
* sélection de la complétion *machine learning*

Le coût de la pression de la touche droite est noté :math:`\delta \infegal 1` qu'on prendra inférieur à 1.
On remarque également qu'avec cette nouvelle métrique, il est possible
de diminuer le nombre minimum de touches à presser pour des requêtes en dehors 
de l'ensemble :math:`S` à partir du moment où elles prolongent une complétion existante.
C'est là un point très intéressant de cette métrique.
De manière évidente, :math:`\forall q, \; M'(q, S) \infegal M"(q, S)`.

Questions
+++++++++

Grâce à cette métrique, on peut envisager de trouver des réponses à certaines questions :

#. Les différences entre les trois métriques sont-elles négligeables ou non ?
#. Ajouter des complétions non présentes dans le corpus améliore-t-elle la métrique ?
   Même question pour la suppression ?
#. Existe-t-il un moyen de construire de façon itérative l'ensemble des complétions
   ou plutôt l'ordre qui minimise la métrice :math:`M'(q, S)` ?
#. Comment calculer rapidement les métriques pour les requêtes dans l'ensemble 
   :math:`S` et en dehors ?
  
Pour la première question, une expérience devrait donner une piste
à défaut d'y répondre. Pour la seconde, il n'est pas nécessaire d'envisager 
la suppression de complétions car celles-ci devraient naturellement se positionner 
en fin de liste. L'ajout correspond à la situation où beaucoup de complétions
partagent le même préfixe sans pour autant que ce préfixe fasse partie de la 
liste des complétions.

::

    macérer
    maline
    machinerie
    machinerie infernale
    machinerie infernalissime
    machine artistique
    machine automatique
    machine chaplin
    machine intelligente
    machine learning
    
L'idée consiste à ajouter la complétion *machine* qui sert de
préfixe commun à beaucoup de complétions et cela améliore le gain moyen
dans le cas présent (sans compter le gain sur la requête
*machine*). Enfin, la troisième et la quatrième question,
la réponse requiert la démonstration de quelques propriétés mathématiques.
Mais avant j'ajouterai que la première métrique :math:`M` correspond 
à la ligne de commande Windows et la métrique :math:`M'` correspond à
la ligne de commande Linux.

