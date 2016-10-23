
=======
k-means
=======

.. index:: centres mobiles, k-means, variance intra-classe, inertie

*Dénomination française : algorithme des centres mobiles.*


Principe
========


Les centres mobiles ou nuées dynamiques sont un algorithme de classification 
*non supervisée*. A partir d'un ensemble de points, il détermine pour un 
nombre de classes fixé, une répartition des points qui minimise un 
critère appelé *inertie* ou variance *intra-classe*.

.. mathdef:: 
    :title: centre mobile, k-means
    :tag: Algorithme
    :lid: hmm_classification_obs_un

    
    On considère un ensemble de points :
    
    .. math::
    
        \left(X_i\right)_{1\leqslant i\leqslant P}\in\left(\R^N\right)^P
    
    A chaque point est associée une classe : 
    :math:`\left(c_i\right)_{1\leqslant i\leqslant P}\in\left\{1,...,C\right\}^P`.
    On définit les barycentres des classes :
    :math:`\left( G_i\right)_{1\leqslant i\leqslant C}\in\left(\R^N\right)^C.
    
    *Initialisation*
    
    L'initialisation consiste à choisir pour chaque point une classe aléatoirement dans 
    :math:`\left\{1,...,C\right\}`. On pose :math:`t = 0`.
    
    .. _hmm_cm_step_bary:
    
    *Calcul des barycentres*
    
    | for k in :math:`1..C`
    |   :math:`G_k^t \longleftarrow \sum_{i=1}^P X_i \, \mathbf{1}_{\left\{c_i^t=k\right\}} \sum_{i=1}^P \mathbf{1}_{\left\{c_i^t=k\right\}}
    
    *Calcul de l'inertie*
    
    .. math::
        
        \begin{array}{lll}
        I^t &\longleftarrow& \sum_{i=1}^P \; d^2\left(X_i, G_{c_i^t}^t\right) \\
        t   &\longleftarrow& t+1
        \end{array}
                
    | if :math:`t > 0` et :math:`I_t \sim I_{t-1}`
    |   arrêt de l'algorithme
    
    .. _hmm_cm_step_attr:
    
    *Attribution des classes*
    
    | for in :math:`1..P`
    |   :math:`c_i^{t+1} \longleftarrow \underset{k}{\arg\min} \; d\left(  X_{i},G_{k}^{t}\right)`
    |   où :math:`d\left(X_i,G_k^t\right)` est la distance entre :math:`X_i` et :math:`G_k^t`
    
    
    Retour à l'étape du calcul des barycentres jusqu'à convergence de l'inertie :math:`I^t`.
    


.. mathdef::
    :title: convergence des k-means
    :tag: Théorème
    :lid: theoreme_inertie_1

    Quelque soit l'initialisation choisie, la suite :math:`\pa{I_t}_{t\supegal 0}`
    construite par l'algorithme des :ref:`k-means <hmm_classification_obs_un>`
    converge.


La démonstration du théorème nécessite le lemme suivant.

.. mathdef::
    :title: inertie minimum
    :tag: Lemme
    :lid: lemme_inertie_minimum

    Soit :math:`\vecteur{X_1}{X_P} \in \pa{\R^N}^P`, 
    :math:`P` points de :math:`\R^N`, le minimum de la quantité 
    :math:`Q\pa{Y \in \R^N}` :
    
    .. math::
        :nowrap:
        
        \begin{eqnarray}
        Q\pa{Y} &=& \sum_{i=1}^P \; d^2\pa{X_i,Y}
        \end{eqnarray}
        
    est atteint pour :math:`Y=G=\dfrac{1}{P} \sum_{i=1}^{P} X_i` 
    le barycentre des points :math:`\vecteur{X_1}{X_P}`.


Soit :math:`\vecteur{X_1}{X_P} \in \pa{\R^N}^P`, 
:math:`P` points de :math:`\R^N`.

.. math::
    :nowrap:

    \begin{eqnarray*}
                        \sum_{i=1}^{P} \overrightarrow{GX_{i}} = \overrightarrow{0}  
    &\Longrightarrow&      \sum_{i=1}^{P} d^2\pa{X_i,Y} = \sum_{i=1}^{P} d^2\pa{X_i,G}+ P \, d^2\pa{G,Y} \\
    &\Longrightarrow&     \underset{Y\in\R^N}{\arg\min} \; \sum_{i=1}^{P} d^2\pa{X_i,Y} = \acc{G}
    \end{eqnarray*}


On peut maintenant démontrer le théorème.
L'étape d'attribution des classes consiste à attribuer à chaque 
point le barycentre le plus proche. On définit :math:`J_t` par :

.. math::
    :nowrap:

    \begin{eqnarray}
    J^{t+1} &=&    \sum_{i=1}^{P} \; d^2\pa{ X_i, G_{c_i^{t+1}}^t} 
    \end{eqnarray}
            
On en déduit que :        

.. math::
    :nowrap:
            
    \begin{eqnarray}
    J^{t+1}    &=& \sum_{i, c_i^t \neq c_i^{t+1}} \; d^2\pa{ X_i, G_{c_i^{t+1}}^t} + J^{t+1} \sum_{i, c_i^t = c_i^{t+1}} \; d^2\pa{ X_i, G_{c_i^{t+1}}^t}  \\
    J^{t+1}    &\infegal&  \sum_{i, c_i^t \neq c_i^{t+1}} \; d^2\pa{ X_i, G_{c_i^{t}}^t} + \sum_{i, c_i^t = c_i^{t+1}} \; d^2\pa{ X_i, G_{c_i^{t}}^t} \\
    J^{t+1}    &\infegal&  I^t
    \end{eqnarray}

Le lemme précédent appliqué à chacune des classes :math:`\ensemble{1}{C}`, 
permet d'affirmer que :math:`I^{t+1} \infegal J^{t+1}`. 
Par conséquent, la suite :math:`\pa{I_t}_{t\supegal 0}` est décroissante et minorée par 
0, elle est donc convergente.

.. index:: convexité


L'algorithme des centres mobiles cherche à attribuer à chaque 
point de l'ensemble une classe parmi les :math:`C` disponibles. 
La solution trouvée dépend de l'initialisation et n'est pas forcément 
celle qui minimise l'inertie intra-classe : l'inertie finale est 
un minimum local. Néanmoins, elle assure que la partition est formée 
de classes convexes : soit :math:`c_1` et :math:`c_2` deux classes différentes, 
on note :math:`C_1` et :math:`C_2` les enveloppes convexes des points qui 
constituent ces deux classes, alors 
:math:`\overset{o}{C_1} \cap \overset{o}{C_2} = \emptyset`. 
La figure suivante présente un exemple d'utilisation de l'algorithme 
des centres mobiles. Des points sont générés aléatoirement 
dans le plan et répartis en quatre groupes.


.. images:: images/cm.png

C'est une application des centres mobiles avec une classification en quatre classes 
d'un ensemble aléatoire de points plus dense sur la partie droite du graphe. Les quatre classes
ainsi formées sont convexes.

.. _hmm_classification_obs_deux:

Homogénéité des dimensions
++++++++++++++++++++++++++

Les coordonnées des points 
:math:`\left(X_i\right) \in \R^N` sont généralement non homogènes : 
les ordres de grandeurs de chaque dimension sont différents. 
C'est pourquoi il est conseillé de centrer et normaliser chaque dimension.
On note : :math:`\forall i \in \intervalle{1}{P}, \; X_i = \vecteur{X_{i,1}}{X_{i,N}}` :

.. math::
    :nowrap:

    \begin{eqnarray*}
    g_k &=& \pa{EX}_k = \frac{1}{P} \sum_{i=1}^P X_{i,k} \\
    v_{kk} &=& \pa{E\left(X-EX\right)^2}_{kk}=\pa{EX^2}_{kk} - g_k^2
    \end{eqnarray*}

Les points centrés et normalisés sont :

.. math::

    \forall i \in \intervalle{1}{P}, \;
    X_i^{\prime}=\left(\dfrac{x_{i,1}-g_{1}}{\sqrt{v_{11}}},...,\dfrac{x_{i,N}-g_{N}}{\sqrt{v_{NN}}}\right)
    
.. index:: Malahanobis 

L'algorithme des centres mobiles est appliqué sur l'ensemble 
:math:`\left( X_{i}^{\prime}\right)_{1\leqslant i\leqslant P}`. 
Il est possible ensuite de décorréler les variables ou d'utiliser 
une distance dite de `Malahanobis <https://fr.wikipedia.org/wiki/Distance_de_Mahalanobis>`_ définie par 
:math:`d_M\pa{X, Y} = X \, M \, Y'` où :math:`Y'` 
désigne la transposée de :math:`Y` et :math:`M` 
est une matrice symétrique définie positive.
Dans le cas de variables corrélées, la matrice 
:math:`M = \Sigma^{-1}` où :math:`\Sigma^{-1}` est la matrice 
de variance-covariance des variables aléatoires :math:`\pa{X_i}_i`.


.. _hmm_classification_obs_trois:


Estimation de probabilités
==========================

A partir de cette classification en :math:`C` classes, on construit un 
vecteur de probabilités pour chaque point :math:`\pa{X_{i}}_{1 \infegal i \infegal P}` 
en supposant que la loi de :math:`X` sachant sa classe :math:`c_X` est une loi 
normale multidimensionnelle. La classe de :math:`X_i` est 
notée :math:`c_i`. On peut alors écrire :

.. math::
    :nowrap:

    \begin{eqnarray}
    \forall i \in \intervalle{1}{C}, \; & & \\
    G_i &=& E\pa{X \indicatrice{c_X = i}} = \dfrac{\sum_{k=1}^{P} X_k \indicatrice {c_k = i}} {\sum_{k=1}^{P} \indicatrice {c_k = i}} \\
    V_i &=& E\pa{XX' \indicatrice{c_X = i}} = \dfrac{\sum_{k=1}^{P} X_k X_k' \indicatrice {c_k = i}} {\sum_{k=1}^{P} \indicatrice {c_k = i}} \\
    \pr{c_X = i} &=& \sum_{k=1}^{P} \indicatrice {c_k = i} \label{hmm_rn_densite_p}\\
    f\pa{X | c_X = i} &=& \dfrac{1}{\pa{2\pi}^{\frac{N}{2}} \sqrt{\det \pa{V_i}}} \; e^{ - \frac{1}{2} \pa{X - G_i}' \; V_i^{-1} \; \pa{X - G_i} } \\
    f\pa{X} &=& \sum_{k=1}^{P}  f\pa{X | c_X = i} \pr{c_X = i} \label{hmm_rn_densite_x}
    \end{eqnarray}

On en déduit que :

.. math::

    \pr{c_X = i |X } = \dfrac{f\pa{X | c_X = i}\pr{c_X = i}} {f\pa{X} }

La densité des obervations est alors modélisée par une mélange de 
lois normales, chacune centrée au barycentre de chaque classe. 
Ces probabilités peuvent également être apprises par un réseau de neurones 
classifieur où servir d'initialisation à un 
`algorithme EM <https://fr.wikipedia.org/wiki/Algorithme_esp%C3%A9rance-maximisation>`_.



Sélection du nombre de classes
==============================

.. _classification_selection_nb_classe_bouldin:

Critère de qualité
++++++++++++++++++

L'algorithme des centres mobiles effectue une classification non supervisée 
à condition de connaître au préalable le nombre de classes et 
cette information est rarement disponible. Une alternative consiste à 
estimer la pertinence des classifications obtenues pour différents 
nombres de classes, le nombre de classes optimal est celui 
qui correspond à la classification la plus pertinente.
Cette pertinence ne peut être estimée de manière unique, elle dépend des 
hypothèses faites sur les éléments à classer, notamment sur la forme 
des classes qui peuvent être convexes ou pas, être modélisées par des 
lois normales multidimensionnelles, à matrice de covariances diagonales, ... 
Les deux critères qui suivent sont adaptés à l'algorithme des centres mobiles. 
Le critère de `Davies-Bouldin <https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index>`_ 
(voir [Davies1979]_) 
est minimum lorsque le nombre de classes est optimal.

.. index:: Davies, Bouldin

.. math::
    :nowrap:

    \begin{eqnarray}
    DB &=& \dfrac{1}{C} \;     \sum_{i=1}^{C} \; \max_{i \neq j} \; \dfrac{\sigma_i + \sigma_j}{ d\pa{C_i,C_j}} 
    \end{eqnarray}
    
Avec :

.. list-table::
    :widths: 5 10
    :header-rows: 1
    
    * - :math:`C`
      - nombre de classes
    * - :math:`\sigma_i`
      - écart-type des distances des observations de la classe :math:`i`
    * - :math:`C_i`
      - centre de la classe :math:`i`

Le critère de `Goodman-Kruskal <https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_gamma>`_
(voir [Goodman1954]_) est quant à lui maximum lorsque le nombre de classes est optimal. 
Il est toutefois plus coûteux à calculer.

.. index:: Goodman, Kruskal

.. math::
    :nowrap:
    
    \begin{eqnarray}
    GK &=& \dfrac{S^+ - S^-} { S^+ + S^-} 
    \end{eqnarray}

Avec :

.. math::
    :nowrap:
    
    \begin{eqnarray*}
    S^+ &=& \acc{ \pa{q,r,s,t} \sac d\pa{q,r} < d\pa{s,t}  \\
    S^- &=& \acc{ \pa{q,r,s,t} \sac d\pa{q,r} < d\pa{s,t} 
    \end{eqnarray}
    
Où :math:`\pa{q,r}` sont dans la même classe et :math:`\pa{s,t}` sont dans des classes différentes.

.. list-table::
    :widths: 10 10
    :header-rows: 0
    
    * - .. image:: images/class_4.png
      - .. image:: images/class_4_db.png  

Classification en quatre classes : nombre de classes sélectionnées par le critère
de Davies-Bouldin dont les valeurs sont illustrées par le graphe apposé à droite.

Maxima de la fonction densité
+++++++++++++++++++++++++++++



L'article [Herbin2001]_ propose une méthode différente pour estimer 
le nombre de classes, il s'agit tout d'abord d'estimer la fonction 
densité du nuage de points qui est une fonction de 
:math:`\R^n \longrightarrow \R`. Cette estimation est effectuée au moyen 
d'une méthode non paramètrique telle que les estimateurs à noyau 
(voir [Silverman1986]_)
Soit :math:`\vecteur{X_1}{X_N}` un nuage de points inclus dans une image, 
on cherche à estimer la densité :math:`f_H\pa{x}` au pixel :math:`x` :

.. math::

    \hat{f}_H\pa{x} = \dfrac{1}{N} \; \sum_{i=1}^{N} \; \dfrac{1}{\det H} \; K\pa{ H^{-1} \pa{x - X_i}} 
    
Où : 

.. math::

    K\pa{x} = \dfrac{1}{ \pa{2 \pi}^{ \frac{d}{2}} } \; e^{ - \frac{ \norme{x}^2 } {2} } 
    
:math:`H` est un paramètre estimée avec la règle de Silverman.
L'exemple utilisé dans cet article est un problème de segmentation 
d'image qui ne peut pas être résolu par la méthode des nuées 
dynamiques puisque la forme des classes n'est pas convexe, 
ainsi que le montre la figure suivante. La fonction de densité 
:math:`f` est seuillée de manière à obtenir une fonction 
:math:`g : \R^n \longrightarrow \acc{0,1}` définie par :

.. math::

    g \pa{x} = \indicatrice{f\pa{x} \supegal s}


.. index:: composante connexe

L'ensemble :math:`g^{-1}\pa{\acc{1}} \subset \R^n` 
est composée de :math:`N` composantes connexes notées 
:math:`\vecteur{C_1}{C_N}`, la classe d'un point :math:`x` 
est alors l'indice de la composante connexe à la 
laquelle il appartient ou la plus proche le cas échéant.

.. list-table::
    :widths: 10 10
    :header-rows: 0
    
    * - .. image:: images/herbin1.png
      - .. image:: images/herbin2.png  
      
Exemple de classification non supervisée appliquée à un problème
de segmentation d'image, la première figure montre la densité obtenue,
la seconde figure illustre la classification obtenue, figure extraite de [Herbin2001]_.
Cette méthode paraît néanmoins difficilement applicable lorsque la 
dimension de l'espace vectoriel atteint de grande valeur. L'exemple de l'image 
est pratique, elle est déjà découpée en région représentées par les pixels, 
l'ensemble :math:`g^{-1}\pa{\acc{1}}` correspond à 
l'ensemble des pixels :math:`x` pour lesquels :math:`f\pa{x} \supegal s`.



Décroissance du nombre de classes
+++++++++++++++++++++++++++++++++


L'article [Kothari1999]_ propose une méthode permettant de 
faire décroître le nombre de classes afin de choisir le nombre 
approprié. L'algorithme des centres mobiles 
proposent de faire décroître l'inertie notée :math:`I` 
définie pour un ensemble de points noté :math:`X = \vecteur{x_1}{x_N}`
et :math:`K` classes. La classe d'un élément :math:`x` 
est notée :math:`C\pa{x}`. Les centres des classes sont notés 
:math:`Y = \vecteur{y_1}{y_K}`. 
L'inertie de ce nuage de points est définie par :

.. math::

    I  =  \sum_{x \in X} \; \norme{ x - y_{C\pa{x} }}^2 
    
On définit tout d'abord une distance 
:math:`\alpha \in \R^+`, puis l'ensemble 
:math:`V\pa{y,\alpha} = \acc{ z \in Y \sac d\pa{y,z} \infegal \alpha }`, 
:math:`V\pa{y,\alpha}` est donc l'ensemble des voisins des 
centres dont la distance avec :math:`y` est inférieur à :math:`\alpha`. 
L'article [Kothari1999]_ propose de minimiser le coût :math:`J\pa{\alpha}`
suivant :

.. math::

    J\pa{\alpha} = \sum_{x \in X} \; \norme{ x - y_{C\pa{x} }}^2 + \sum_{x \in X} \; 
    \sum_{y \in V\pa{y_{C\pa{x}}, \alpha} } \; \lambda\pa{y} \, \norme{ y -  y_{C\pa{x}}}^2

Lorsque :math:`\alpha` est nul, ce facteur est égal à l'inertie : 
:math:`I = J\pa{0}` et ce terme est minimal lorsqu'il y a autant de 
classes que d'éléments dans :math:`X`. Lorsque :math:`\alpha` 
tend vers l'infini, :math:`J\pa{\alpha} \rightarrow J\pa{\infty}` où :

.. math::

    J\pa{\infty} = \sum_{x \in X} \; \norme{ x - y_{C\pa{x} }}^2 + \sum_{x \in X} \; \sum_{y \in Y} \; 
    \lambda\pa{y} \, \norme{ y -  y_{C\pa{x}}} ^2

Ici encore, il est possible de montrer que ce terme 
:math:`J\pa{\infty}` est minimal lorsqu'il n'existe plus qu'une 
seule classe. Le principe de cette méthode consiste à faire varier 
le paramètre :math:`\alpha`, plus le paramètre :math:`\alpha` augmente, 
plus le nombre de classes devra être réduit. Néanmoins, il existe 
des intervalles pour lequel ce nombre de classes est stable, 
le véritable nombre de classes de l'ensemble :math:`X` 
sera considéré comme celui correspondant au plus grand intervalle 
stable.


.. list-table::
    :widths: 10 10
    :header-rows: 0
    
    * - .. image:: images/koth1.png
      - .. image:: images/koth2.png  
    * - *(a)*
      - *(b)*
      
Evolutation du nombre de classes en fonction du paramètre :math:`\alpha` lors de la 
minimisation du critère :math:`J\pa{\alpha}`, figure extraite de [Kothari1999]_.
La première image représente le nuage de points illustrant quatre classes sans recouvrement.
La seconde image montre que quatre classes est l'état le plus longtemps stable
lorsque :math:`\alpha` croît.


.. index:: multiplicateurs de Lagrange

Le coût :math:`J\pa{\alpha}` est une somme de coût dont 
l'importance de l'un par rapport à l'autre est contrôle 
par les paramètres :math:`\lambda\pa{y}`. Le problème de 
minimisation de :math:`J\pa{\alpha}` est résolu par l'algorithme qui suit. 
Il s'appuie sur la méthode des multiplicateurs de Lagrange.

.. mathdef::
    :title: sélection du nombre de classes (Kothari1999)
    :tag: Algorithme
    :lid: classification_kothari_1999

    Les notations sont celles utilisés dans les paragraphes précédents. On suppose que le 
    paramètre :math:`\alpha` évolue dans l'intervalle :math:`\cro{\alpha_1, \alpha_2}` 
    à intervalle régulier :math:`\alpha_t`.
    Le nombre initial de classes est noté :math:`K` et il est supposé surestimer le véritable 
    nombre de classes. Soit :math:`\eta \in \left]0,1\right[`, 
    ce paramètre doit être choisi de telle sorte que dans
    l'algorithme qui suit, l'évolution des centres :math:`y_k` 
    soit autant assurée par le premier de la fonction de coût que par le second.
    
    *initialisation*
    
    .. math::
    
        \alpha \longleftarrow \alpha_1
        
    On tire aléatoirement les centres des :math:`K` classes :math:`\vecteur{y_1}{y_K}`.
    
    *préparation*
    
    On définit les deux suites entières :math:`\vecteur{c^1_1}{c^1_K}`, :math:`\vecteur{c^2_1}{c^2_K}`, 
    et les deux suites de vecteur :math:`\vecteur{z^1_1}{z^1_K}`, 
    :math:`\vecteur{z^2_1}{z^2_K}`.
    
    .. math::
        
        \begin{array}{rlll}
        \forall k, &  c^1_k &=& 0 \\ 
        \forall k, &  c^2_k &=& 0 \\ 
        \forall k, &  z^1_k &=& 0 \\ 
        \forall k, &  z^2_k &=& 0 
        \end{array}

    *calcul des mises à jour*
    
    | for i in :math:`1..N`
    |   Mise à jour d'après le premier terme de la fonction de coût :math:`J\pa{\alpha}`.
    |
    |   .. math::
    |
    |       \begin{array}{lll}
    |       w             &\longleftarrow&         \underset{1 \infegal l \infegal K}{\arg \min} \; \norme{x_i - y_l}^2 \\
    |       z^1_w     &\longleftarrow&         z^1_w + \eta \pa{ x_i - y_w} \\
    |       c^1_w     &\longleftarrow&         c^1_w + 1 
    |       \end{array}
    | 
    |   Mise à jour d'après le second terme de la fonction de coût :math:`J\pa{\alpha}`
    |
    |   for v in :math:`1..k`
    |       if :math:`\norme{y_v - y_w} < \alpha`
    |
    |           .. math::
    |
    |               \begin{array}{lll}
    |               z^2_v     &\longleftarrow&         z^2_v - \pa{ y_v - y_w} \\
    |               c^2_v     &\longleftarrow&         c^2_v + 1 
    |               \end{array}
    |
    |   for v in :math:`1..k`
    |
    |       .. math::
    |
    |           \begin{array}{lll}
    |           \lambda_v &\longleftarrow& \frac{ c^2_v \norme{z^1_v} } { c^1_v \norme{z^2_v} } \\
    |           y_v                &\longleftarrow& y_v + z^1_v + \lambda_v z^2_v
    |           \end{array}

    *convergence*
    
    Tant que l'étape précédente n'a pas convergé vers une version stable des centres,
    :math:`y_k`, retour à l'étape précédente. Sinon, tous les couples de classes :math:`\pa{i,j}` 
    vérifiant :math:`\norme{y_i - y_j} > \alpha` sont fusionnés :
    :math:`\alpha \longleftarrow \alpha + \alpha_t`.
    Si :math:`\alpha \infegal \alpha2`, retour à l'étape de préparation.
    
    *terminaison*
    
    Le nombre de classes est celui ayant prévalu pour le plus grand nombre de valeur de :math:`\alpha`.



Extension des nuées dynamiques
==============================

.. _classification_nuees_dynamique_extension:

Classes elliptiques
+++++++++++++++++++

.. index:: classes elliptiques


La version de l'algorithme des nuées dynamique proposée dans l'article 
[Cheung2003]_ suppose que les classes ne sont plus de forme circulaire 
mais suivent une loi normale quelconque. La loi de l'échantillon 
constituant le nuage de points est de la forme :

.. math::

    f\pa{x} =  \sum_{i=1}^{N} \; p_i \; \dfrac{1}{\pa{2 \pi}^{\frac{d}{2}}\sqrt{\det \Sigma_i}} \; exp \pa{-\frac{1}{2}  \pa{x-\mu_i}' \Sigma_i^{-1} \pa{x-\mu_i} } 
    \end{eqnarray}

Avec :math:`sum_{i=1}^{N} \; p_i = 1`. On définit :

.. math::

    G\pa{x, \mu, \Sigma} = \dfrac{1}{\pa{2 \pi}^{\frac{d}{2}}\sqrt{\det \Sigma}} \; exp \pa{-\frac{1}{2}  \pa{x-\mu}' \Sigma^{-1} \pa{x-\mu} }

L'algorithme qui suit a pour objectif de minimiser la quantité pour un échantillon :math:`\vecteur{X_1}{X_K}` :

.. math::
    :nowrap:
    
    \begin{eqnarray}
    I = \sum_{i=1}^{N}\sum_{k=1}^{K} \indicatrice{ i = \underset{1 \infegal j \infegal N}{\arg \max} 
    G\pa{X_k, \mu_j,\Sigma_j} } \; \ln \cro{ p_i G\pa{ X_k, \mu_i, \Sigma_i } }
    \end{eqnarray}


.. mathdef::
    :title: nuées dynamiques généralisées
    :tag: Algorithme
    
    Les notations sont celles utilisées dans ce paragraphe. Soient :math:`\eta`, 
    :math:`\eta_s` deux réels tels que :math:`\eta > \eta_s`. 
    La règle préconisée par l'article [Cheung2003]_ est :math:`\eta_s \sim \frac{\eta}{10}`.
    
    *initialisation*
    
    :math:`t \longleftarrow 0`.
    Les paramètres :math:`\acc{p_i^0, \mu_i^0, \Sigma_i^0 \sac 1 \infegal i \infegal N}` sont initialisés
    grâce à un algorithme des :ref:`k-means <hmm_classification_obs_un>` ou :ref:`FSCL <label_kmeans_fscl>`.
    :math:`\forall i, \; p_i^0 = \frac{1}{N}` et :math:`\beta_i^0 = 0`.
    
    *récurrence*
    
    Soit :math:`X_k` choisi aléatoirement dans :math:`\vecteur{X_1}{X_K}`.
    
    .. math::
    
        i = \underset{1 \infegal i \infegal N}{\arg \min} \; G\pa{X_k, \mu_i^t, \Sigma_i^t}
        
    | for i in :math:`1..N`
    |
    |   .. math::
    |
    |       \begin{array}{lll}
    |       \mu_i^{t+1}         &=& \mu_i^t + \eta \, \pa{\Sigma_i^t}^{-1} \, \pa{ X_k - \mu_i^t} \\
    |       \beta_i^{t+1}     &=& \beta_i^t + \eta \, \pa{1 - \alpha_i^t} \\
    |       \Sigma^{t+1}_i     &=& \pa{1 - \eta_s} \, \Sigma_i^t + \eta_s \, \pa{ X_k - \mu_i^t} \pa{ X_k - \mu_i^t}'
    |       \end{array}
    |
    | for i in :math:`1..N`
    |   :math:`p^{t+1}_i = \frac{ e^{ \beta_i^{t+1} } } { \sum_{j=1}^{N} e^{ \beta_j^{t+1} } }`
    |
    | :math:`t \longleftarrow t + 1`

    *terminaison*
    
    Tant que :math:`\underset{1 \infegal i \infegal N}{\arg \min} \; G\pa{X_k, \mu_i^t, \Sigma_i^t}`
    change pour au moins un des points :math:`X_k`.
            
Lors de la mise à jour de :math:`\Sigma^{-1}`,
l'algorithme précédent propose la mise à jour de :math:`\Sigma_i` 
alors que le calcul de :math:`G\pa{., \mu_i, \Sigma_i}` 
implique :math:`\Sigma_i^{-1}`, 
par conséquent, il est préférable de mettre à jour directement la matrice 
:math:`\Sigma^{-1}` :

.. math::

    \pa{\Sigma^{t+1}_i}^{-1} = \frac{ \pa{\Sigma_i^t}^{-1} } {1 - \eta_s} 
    \cro{I - \frac{ \eta_s  \pa{ X_k - \mu_i^t} \pa{ X_k - \mu_i^t}' \pa{\Sigma_i^t}^{-1} }
    {1 - \eta_s + \eta_s \pa{ X_k - \mu_i^t}' \, \pa{\Sigma_i^t}^{-1}\pa{ X_k - \mu_i^t} } }


.. _class_rpcl:

Rival Penalized Competitive Learning (RPCL)
+++++++++++++++++++++++++++++++++++++++++++

.. index:: Rival Penalized Competitive Learning, RPCL


L'algorithme suivant développé dans [Xu1993]_, est une variante de celui des centres mobiles. 
Il entreprend à la fois la classification et la sélection du nombre optimal de classes à condition 
qu'il soit inférieur à une valeur maximale à déterminer au départ de l'algorithme.
Un mécanisme permet d'éloigner les centres des classes peu pertinentes 
de sorte qu'aucun point ne leur sera affecté.

.. mathdef::
    :title: RPCL
    :tag: Algorithme
    :lid: classif_algo_rpcl
    
    Soient :math:`\vecteur{X_1}{X_N}`, :math:`N` vecteurs à classer en au 
    plus :math:`T` classes de centres :math:`\vecteur{C_1}{C_T}`. 
    Soient deux réels :math:`\alpha_r` et :math:`\alpha_c` 
    tels que :math:`0 < \alpha_r \ll \alpha_c < 1`.

    *initialisation*
    
    Tirer aléatoirement les centres :math:`\vecteur{C_1}{C_T}`.
    
    | for j in :math:`1..C`
    |   :math:`n_j^0 \longleftarrow 1`

    *calcul de poids*
    
    Choisir aléatoirement un point :math:`X_i`.
    
    | for j in :math:`1..C`
    |   :math:`\gamma_j = \dfrac{n_j}{ \sum_{k=1}^{C} n_k`
    |
    | for j in :math:`1..C`
    |
    |   .. math::
    |
    |       u_j = \left \{ \begin{array}{ll} 1   & \text{si} j \in \underset{k}{\arg \min} \; \cro {\gamma_k \; d\pa{X_i,C_k} } \\
    |       -1  & \text{si} j \in \underset{j \neq k}{\arg \min} \; \cro {\gamma_k \; d\pa{X_i,C_k} } \\
    |       0   & \text{sinon}
    |       \end{array} \right.
            
    *mise à jour*
    
    | for j in :math:`1..C`
    | 
    |   .. math::
    |
    |       \begin{array}{lcl}
    |       C_j^{t+1} &\longleftarrow&  C_j^t +  \left \{ \begin{array}{ll} \alpha_c \pa{X_i - C_j} & \text{si } u_j = 1 \\
    |       - \alpha_r \pa{X_i - C_j} & \text{si } u_j = -1 \\ 0 & \text{sinon} \end{array} \right. \\
    |       n_j^{t+1} &\longleftarrow&  n_j^t +  \left \{ \begin{array}{ll} 1 & \text{si } u_j = 1 \\
    |       0 & \text{sinon} \end{array} \right. 
    |
    | :math:`t \longleftarrow t+1`
    
    *terminaison*
    
    S'il existe un indice :math:`j` pour lequel :math:`C^{t+1}_j \neq C^t_j` 
    alors retourner à  l'étape de calcul de poids ou que les centres des classes jugées inutiles 
    ont été repoussés vers l'infini.


Pour chaque point, le centre de la classe la plus proche en est rapproché 
tandis que le centre de la seconde classe la plus proche en est éloigné 
mais d'une façon moins importante (condition :math:`\alpha_r \ll \alpha_c`). 
Après convergence, les centres des classes inutiles ou non pertinentes 
seront repoussés vers l'infini. Par conséquent, aucun point n'y sera rattaché.

L'algorithme doit être lancé plusieurs fois. L'algorithme RPCL peut terminer 
sur un résultat comme celui de la figure suivante où un centre reste coincé 
entre plusieurs autres. Ce problème est moins important 
lorsque la dimension de l'espace est plus grande.

.. image:: images/class6.ong

Application de l'algorithme :ref:`RPCL <classif_algo_rpcl>` : la classe 0 est incrusté entre les quatre autres 
et son centre ne peut se "faufiler" vers l'infini.


.. _classification_rpcl_local_pca:

RPCL-based local PCA
++++++++++++++++++++

.. index:: RPCL, PCA, ellipse


L'article [Liu2003]_ propose une extension de l'algorithme :ref:`RPCL <classif_algo_rpcl>` 
et suppose que les classes ne sont plus de forme circulaire mais 
suivent une loi normale quelconque. Cette méthode est utilisée pour 
la détection de ligne considérées ici comme des lois normales dégénérées 
en deux dimensions, la matrice de covariance définit une ellipse dont le 
grand axe est très supérieur au petit axe, ce que montre la figure suivante. 
Cette méthode est aussi présentée comme un possible algorithme de squelettisation.

.. image:: images/liu3.png

Figure extraite de [Liu2003]_, l'algorithme est utilisé pour la détection de lignes
considérées ici comme des lois normales dont la matrice de covariance définit une ellipse
dégénérée dont le petit axe est très inférieur au grand axe. Les traits fin grisés correspondent aux 
classes isolées par l'algorithme RPCL-based local PCA.

On modélise le nuage de points par une mélange de lois normales :

.. math::

    f\pa{x} =  \sum_{i=1}^{N} \; p_i \; \dfrac{1}{\pa{2 \pi}^{\frac{d}{2}}\sqrt{\det \Sigma_i}} \;
    exp \pa{-\frac{1}{2}  \pa{x-\mu_i}' \Sigma_i^{-1} \pa{x-\mu_i} } 
    
Avec :math:`\sum_{i=1}^{N} \; p_i = 1`.

On suppose que le nombre de classes initiales :math:`N` surestime le 
véritable nombre de classes. L'article [Liu2003]_ s'intéresse 
au cas particulier où les matrices de covariances vérifient
:math:`\Sigma_i = \zeta_i \, I + \sigma_i \, \phi_i \phi_i'`
avec :math:`\zeta_i > 0, \; \sigma_i > 0, \; \phi_i' \phi_i = 1`.

On définit également :

.. math::

    G\pa{x, \mu, \Sigma} = \dfrac{1}{\pa{2 \pi}^{\frac{d}{2}}\sqrt{\det \Sigma}} \;
    exp \pa{-\frac{1}{2}  \pa{x-\mu}' \Sigma^{-1} \pa{x-\mu} }

L'algorithme utilisé est similaire à l'algortihme :ref:`RPCL <classif_algo_rpcl>`. 
La distance :math:`d` utilisée lors de l'étape de calcul des poids
afin de trouver la classe la plus probable pour un point 
donné :math:`X_k` est remplacée par l'expression :

.. math::

    d\pa{X_k, classe \, i} = - \ln { p_i^t \, G\pa{X_k, \, \mu_i^t, \, \Sigma^t_i } }
                
L'étape de mise à jour des coefficients est remplacée par :

.. math::

    x^{t+1} \longleftarrow  x^t +  \left \{ \begin{array}{ll}
    \alpha_c \nabla x^t & \text{si } u_j = 1 \\
    - \alpha_r \nabla x^t & \text{si } u_j = -1 \\
    0 & \text{sinon}
    \end{array} \right.

Où :math:`x^t` joue le rôle d'un paramètre et est remplacé 
successivement par :math:`p_i^t`, :math:`\mu_i^t`, :math:`\zeta_i^t`, :math:`\sigma^t_i`, :math:`\phi^t_i` :

.. math::

    \begin{array}{lll}
    \nabla p_i^t &=& - \frac{1}{p_i^t} \\
    \nabla \mu_i^t &=& - \pa{ X_k - \mu_i^t} \\
    \nabla \zeta_i^t  &=& \frac{1}{2} \; tr\cro{ \pa{\Sigma_i^t}^{-1} \, 
    \pa{ I - \pa{ X_k - \mu_i^t} \pa{ X_k - \mu_i^t}' \pa{\Sigma_i^t}^{-1} } } \\
    \nabla \sigma_i^t &=&    \frac{1}{2} \; \pa{\phi_i^t}' \pa{\Sigma_i^t}^{-1} 
    \pa{ I - \pa{ X_k - \mu_i^t} \pa{ X_k - \mu_i^t}' \pa{\Sigma_i^t}^{-1} } \phi_i^t \\
    \nabla \phi_i^t     &=&    \sigma_i^t \pa{\Sigma_i^t}^{-1} 
    \pa{ I - \pa{ X_k - \mu_i^t} \pa{ X_k - \mu_i^t}' \pa{\Sigma_i^t}^{-1} } \phi_i^t \\
    \end{array}


.. _label_kmeans_fscl:

Frequency Sensitive Competitive Learning (FSCL)
+++++++++++++++++++++++++++++++++++++++++++++++

.. index:: FSCL, Kohonen


L'algorithme Frequency Sensitive Competitive Learning est présenté dans 
[Balakrishnan1996]_. Par rapport à l'algorithme des centres mobiles classique, 
lors de l'estimation des centres des classes, l'algorithme évite la formation de classes sous-représentées.

.. mathdef::
    :title: FSCL
    :lid: classification_fscl
    :tag: Algorithme
    
    Soit un nuage de points :math:`\vecteur{X_1}{X_N}`, 
    soit :math:`C` vecteurs :math:`\vecteur{\omega_1}{\omega_C}` 
    initialisés de manière aléatoires. 
    Soit :math:`F : \pa{u,t} \in \R^2 \longrightarrow \R^+` 
    croissante par rapport à :math:`u`.
    Soit une suite de réels :math:`\vecteur{u_1}{u_C}`, 
    soit une suite :math:`\epsilon\pa{t} \in \cro{0,1}` décroissante où :math:`t` 
    représente le nombre d'itérations.
    Au début :math:`t \leftarrow 0`.
    
    *meilleur candidat*
    
    Pour un vecteur :math:`X_k` choisi aléatoirement dans 
    l'ensemble :math:`\vecteur{X_1}{X_N}`, on détermine :
    
    .. math::
    
        i^* \in \arg \min \acc{ D_i = F\pa{u_i,t} \, d\pa{X_k, \omega_i} }

    *mise à jour*
    
    | :math:`\omega_{i^*} \pa{t+1}  \longleftarrow \omega_{i^*} \pa{t} + \epsilon\pa{t} \pa { X_k - \omega_{i^*} \pa{t} }`
    | :math:`t \longleftarrow t+1`
    | :math:`u_{i^*} \longleftarrow u_{i^*} + 1`
    
    Retour à l'étape précédente jusqu'à ce que les nombres 
    :math:`\frac{u_i}{\sum_{i}u_i}` convergent.

Exemple de fonctions pour :math:`F`, :math:`\epsilon` (voir [Balakrishnan1996]_) :

.. math::
    :nowrap:
    
    \begin{eqnarray*}
    F\pa{u,t} &=& u \, \beta e^{-t/T}                 \text{ avec } \beta = 0,06 \text{ et } 1/T = 0,00005 \\
    \epsilon\pa{t} &=& \beta \, e^{ - \gamma t } \text{ avec } \gamma = 0,05
    \end{eqnarray*}

Cet algorithme ressemble à celui des cartes topographiques de Kohonen 
sans toutefois utiliser un maillage entre les neurones 
(ici les vecteurs :math:`\omega_i`). Contrairement à l'algorithme RPCL, 
les neurones ne sont pas repoussés s'ils ne sont pas choisis mais la fonction 
croissante :math:`F\pa{u,t}` par rapport à :math:`u` assure que plus un neurone 
est sélectionné, moins il a de chance de l'être, 
bien que cet avantage disparaisse au fur et à mesure des itérations.


Bibliographie
=============

.. [Balakrishnan1996] Comparative performance of the FSCL neural net and K-means algorithm for market segmentation (1996),
   P. V. Sundar Balakrishnan, Martha Cooper, Varghese S. Jacob, Phillip A. Lewis,
   *European Journal of Operation Research*, volume 93, pages 346-357

.. [Cheung2003] :math:`k^*`-Means: A new generalized :math:`kè -means clustering algorithm (2003),
   Yiu-Ming Cheung, 
   *Pattern Recognition Letters*, volume 24, 2883-2893

.. [Davies1979] A cluster Separation Measure (1979),
   D. L. Davies, D. W. Bouldin,
   *IEEE Trans. Pattern Analysis and Machine Intelligence (PAMI)*, volume 1(2)

.. [Goodman1954] Measures of associations for cross-validations (1954),
   L. Goodman, W. Kruskal,
   *J. Am. Stat. Assoc.*, volume 49, pages 732-764

.. [Herbin2001] Estimation of the number of clusters and influence zones (2001),
   M. Herbin, N. Bonnet, P. Vautrot,
   *Pattern Recognition Letters*, volume 22, pages 1557-1568

.. [Kothari1999] On finding the number of clusters (1999),
   Ravi Kothari, Dax Pitts,
   *Pattern Recognition Letters*, volume 20, pages 405-416

.. [Liu2003] Strip line detection and thinning by RPCL-based local PCA (2003),
   Zhi-Yong Liu, Kai-Chun Chiu, Lei Xu,
   *Pattern Recognition Letters* volume 24, pages 2335-2344

.. [Silverman1986] Density Estimation for Statistics and Data Analysis (1986),
   B. W. Silverman,
   *Monographs on Statistics and Applied Probability, Chapman and Hall, London*, volume 26

.. [Xu1993] Rival penalized competitive learning for clustering analysis, rbf net and curve detection (1993),
   L. Xu, A. Krzyzak, E. Oja,
   *IEEE Trans. Neural Networks*, volume (4), pages 636-649


