
.. index:: Levenshtein, Levenstein, distance d'édition, édition

==================
Distance d'édition
==================

Les distances d'édition permettent de comparer deux mots entre eux ou plus 
généralement deux séquences de symboles entre elles. L'usage le 
plus simple est de trouver, pour un mot mal orthographié, le mot le plus 
proche dans un dictionnaire, c'est une option proposée dans la plupart 
des traitements de texte. La distance présentée est la 
`distance de Levenshtein <https://fr.wikipedia.org/wiki/Distance_de_Levenshtein>`_
(voir [Levenstein1966]_)
Elle est parfois appelée 
`Damerau Levenstein Matching (DLM) <https://fr.wikipedia.org/wiki/Distance_de_Damerau-Levenshtein>`_
(voir également [Damerau1964]_). Cette distance fait intervenir trois opérations élémentaires :

* comparaison entre deux caractères
* insertion d'un caractère
* suppression d'un caractère

Pour comparer deux mots, il faut construire une méthode associant 
ces trois opérations afin que le premier mot se transforme en le 
second mot. L'exemple suivant utilise les mots ``idstzance`` et ``distances``, 
il montre une méthode permettant de passer du premier au second. 
La distance sera la somme des coûts associés à chacune des opérations 
choisies. La comparaison entre deux lettres identiques est en général 
de coût nul, toute autre opération étant de coût strictement positif. 


.. list-table:: distance d'édition
   :widths: 3 3 10 3
   :header-rows: 1

    * - mot 1
      - mot 2
      - opération
      - coût
    * - i
      - d
      - comparaison entre ``i`` et ``d``
      - 1
    * - d 
      - i
      - comparaison entre ``d`` et ``i``
      - 1
    * - s
      - s
      - comparaison entre ``s`` et ``s``
      - 0
    * - t
      - t
      - comparaison entre ``t`` et ``t``
      - 0
    * - z 
      -  
      - suppression de ``z``
      - 1
    * - a 
      - a 
      - comparaison entre ``a`` et ``a``
      - 0
    * - n 
      - n 
      - comparaison entre ``n`` et ``n``
      - 0
    * - c 
      - c 
      - comparaison entre ``c`` et ``c``
      - 0
    * - e 
      - e 
      - comparaison entre ``e`` et ``e``
      - 0
    * - 
      - s 
      - insertion de ``s``
      - 1
    * - 
      -  
      - somme
      - 4

Pour cette distance d'édition entre les mots ``idstzance`` et ``distances``.
La succession d'opérations proposée n'est pas la seule qui permettent 
de construire le second mot à partir du premier mais c'est la moins coûteuse.



Définition et propriétés
========================

.. index:: séquence

Définition
++++++++++


Tout d'abord, il faut définir ce qu'est un mot ou une séquence :

.. mathdef::
    :tag: Définition
    :title: mot
    :lid: definition_edit_mot

    On note :math:`\mathcal{C}` l'espace des caractères ou des symboles. Un mot ou une séquence est 
    une suite finie de :math:`\mathcal{C}`. On note
    :math:`\mathcal{S}_\mathcal{C} = \union{k=1}{\infty} C^k` l'espace des mots formés 
    de caractères appartenant à :math:`\mathcal{C}`.

On peut définir la distance d'édition :

.. mathdef::
    :tag: Définition
    :title: distance d'édition
    :lid: defition_distance_edition_1

    La distance d'édition :math:`d` sur :math:`\mathcal{S}_\mathcal{C}` est définie par :
    
    .. math::
    
        \begin{array}{crcl}
        d : & \mathcal{S}_\mathcal{C} \times \mathcal{S}_\mathcal{C} & \longrightarrow & \R^+\\
        & \pa{m_1,m_2} & \longrightarrow & \underset{ \begin{subarray} OO \text{ séquence} \\ \text{d'opérations} \end{subarray}}{ \min} \, d\pa{m_1,m_2,O}
        \end{array}
    

La distance est le coût de la transformation du mot :math:`m_1` en :math:`m_2` la moins coûteuse. 
Il reste à démontrer que cette distance en est bien une
puis à proposer une méthode de calcul plus rapide que celle suggérée par cette définition.


.. index:: edit_demonstration

Propriétés
++++++++++

Ce paragraphe a pour objectif de démontrer que la 
:ref:`distance <defition_distance_edition_1>` en est bien une.

.. mathdef::
    :tag: Définition
    :title: distance entre caractères
    :lid: edition_distance_definition_1
    
    Soit :math:`\mathcal{C}' = \mathcal{C} \bigcup \accolade{.}` 
    l'ensemble des caractères ajouté au caractère vide ``.``.
    On note :math:`c : \pa{\mathcal{C}'}^2 \longrightarrow \R^+` 
    la fonction coût définie comme suit :
    
    .. math::
        :nowrap:
        :eq: equation_edit_car
        
        \begin{eqnarray}
        \forall \pa{x,y} \in \pa{\mathcal{C}'}^2, \; c\pa{x,y} \text{ est le coût } \left\{
        \begin{array}{ll}
        \text { d'une comparaison}  & \text{si } \pa{x,y} \in \pa{\mathcal{C}}^2\\
        \text { d'une insertion}		& \text{si } \pa{x,y} \in \pa{\mathcal{C}} \times \accolade{.}\\
        \text { d'une suppression} 	& \text{si } \pa{x,y} \in \accolade {.} \times \pa{\mathcal{C}} \\
        0 													& \text{si } \pa{x,y} = \pa{\acc{.},\acc{.}}
        \end{array}
        \right.
        \end{eqnarray}
        
    On note :math:`\mathcal{S}_\mathcal{C'}^2 = \union{n=1}{\infty} \pa{\mathcal{C'}^2}^n` 
    l'ensemble des suites finies de :math:`\mathcal{C'}`.


Pour modéliser les transformations d'un mot vers un autre, on définit pour un mot :math:`m` un 
*mot acceptable* :

.. mathdef::
    :tag: Définition
    :title: mot acceptable
    :lid: edition_distance_mot_acceptable_1
    
    Soit :math:`m = \vecteur{m_1}{m_n}` un mot tel qu'il est défini précédemment. 
    Soit :math:`M=\pa{M_i}_{i \supegal 1}` une suite infinie de caractères, on dit que 
    :math:`M` est un mot acceptable pour :math:`m` si et seulement si la sous-suite
    extraite de :math:`M` contenant tous les caractères différents de :math:`\acc{.}` 
    est égal au mot :math:`m`. On note :math:`acc\pa{m}` 
    l'ensemble des mots acceptables pour le mot :math:`m`.


Par conséquent, tout mot acceptable :math:`m'` pour le mot :math:`m` 
est égal à :math:`m` si on supprime les caractères :math:`\acc{.}` 
du mot :math:`m'`. En particulier, à partir d'un certain indice, :math:`m'` 
est une suite infinie de caractères :math:`\acc{.}`. Il reste 
alors à exprimer la définition de la distance d'édition
en utilisant les mots acceptables :

.. mathdef::
    :tag: Définition
    :title: distance d'édition
    :lid: defition_distance_edition_2
    
    Soit :math:`c` la :ref:`distance d'édition <edition_distance_definition_1>`, 
    :math:`d` définie sur :math:`\mathcal{S}_\mathcal{C}` est définie par :
    
    .. math::
        :nowrap:
        :label: equation_edit_mot 
        
        \begin{eqnarray}
        \begin{array}{crcl}
        d : & \mathcal{S}_\mathcal{C} \times \mathcal{S}_\mathcal{C} & \longrightarrow & \R^+\\
            & \pa{m_1,m_2} & \longrightarrow &
                            \min \acc{  \sum_{i=1}^{+\infty} c\pa{M_1^i, M_2^i} |
                                        \pa{M_1,M_2} \in acc\pa{m_1} \times acc\pa{m_2}}
        \end{array}
        \end{eqnarray}

Il est évident que la série :math:`\sum_{i=1}^{+\infty} c\pa{M_1^i, M_2^i}` 
est convergente. La :ref`distance de caractères <edition_distance_definition_1>` implique 
que les distance d'édition définies en :ref:`1 <defition_distance_edition_1>` 
et :ref:`2 <defition_distance_edition_2>` sont identiques.


.. mathdef::
    :tag: Théorème
    :title: distance d'édition
    :lid: edition_distance_theoreme001
    
    Soit :math:`c` et :math:`d` les fonctions définies respectivement par 
    :eq:`equation_edit_car` et :eq:`equation_edit_mot`, alors :
        
        :math:`c` est une distance sur :math:`\mathcal{C} \Longleftrightarrow d`
        est une distance sur :math:`\mathcal{S}_\mathcal{C}`


On cherche d'abord à démontrer que 

    :math:`c` est une distance sur :math:`\mathcal{C}' \Longleftarrow d` 
    est une distance sur :math:`\mathcal{S}_\mathcal{C}`

Cette assertion est évidente car, si :math:`\pa{m_1,m_2}` sont deux mots de un caractère, 
la distance :math:`d` sur :math:`\mathcal{S}_\mathcal{C}` 
définit alors la distance :math:`c` sur :math:`\mathcal{C}'`.


On démontre ensuite que :

    :math:`c` est une distance sur :math:`\mathcal{C}' \Longrightarrow d` 
    est une distance sur :math:`\mathcal{S}_\mathcal{C}`

Soient deux mots :math:`\pa{m_1,m_2}`, 
soit :math:`\pa{M_1,M_2} \in acc\pa{m_1} \times acc\pa{m_2}`, 
comme :math:`c` est une distance sur :math:`\mathcal{C}'` alors
:math:`d\pa{M_1,M_2} = d\pa{M_2,M_1}`.

D'où, d'après la définition :ref:`2 <defition_distance_edition_2>` :

.. math::
    :label: edit_demo_eq_1

    d\pa{m_1,m_2} = d\pa{m_2,m_1} 

Soit :math:`\pa{N_1,N_2} \in acc\pa{m_1} \times acc\pa{m_2}` 
tels que :math:`d\pa{m_1,m_2} = d\pa{N_2,N_1}` alors :

.. math::
    :nowrap:
    :label: edit_demo_eq_2

    \begin{eqnarray*}
    d\pa{m_1,m_2} = 0   & \Longrightarrow &     d\pa{N_1,N_2} = 0 \\
                        & \Longrightarrow &     \summy{i=1}{+\infty} c\pa{N_1^i, N_2^i} = 0 \\
                        & \Longrightarrow &     \forall i \supegal 1, \; N_1^i = N_2^i \\
                        & \Longrightarrow &     N_1 = N_2 \\
    d\pa{m_1,m_2} = 0   & \Longrightarrow &     m_1 = m_2 
    \end{eqnarray*}

Il reste à démontrer l'inégalité triangulaire. 
Soient trois mots :math:`\pa{m_1,m_2,m_3}`, 
on veut démontrer que 
:math:`d\pa{m_1,m_3} \infegal d\pa{m_1,m_2} + d \pa{m_2,m_3}`.
On définit :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \pa{N_1,N_2} \in acc\pa{m_1} \times acc\pa{m_2}    & \text{ tels que }     &  d\pa{m_1,m_2} = d\pa{N_1,N_2} \\
    \pa{P_2,P_3} \in acc\pa{m_2} \times acc\pa{m_3}    & \text{ tels que }     &  d\pa{m_2,m_3} = d\pa{P_2,P_3} \\
    \pa{O_1,O_3} \in acc\pa{m_1} \times acc\pa{m_3}    & \text{ tels que }     &  d\pa{m_1,m_3} = d\pa{O_1,O_3}
    \end{eqnarray*}

Mais il est possible, d'après la définition d'un :ref:`mot acceptable <edition_distance_mot_acceptable_1>` 
d'insérer des caractères :math:`\acc{.}` dans les mots :math:`N_1,N_2,P_2,P_3,O_1,O_3` 
de telle sorte qu'il existe 
:math:`\pa{M_1,M_2,M_3} \in acc\pa{m_1} \times \in acc\pa{m_2} \times \in acc\pa{m_3}` 
tels que :

.. math::
    :nowrap:

    \begin{eqnarray*}
    d\pa{m_1,m_2} = d\pa{M_1,M_2} \\
    d\pa{m_2,m_3} = d\pa{M_2,M_3} \\
    d\pa{m_1,m_3} = d\pa{M_1,M_3}
    \end{eqnarray*}

Or comme la fonction :math:`c` est une distance sur :math:`\mathcal{C}'`, on peut affirmer que :
:math:`d\pa{M_1,M_3} \infegal d\pa{M_1,M_2} + d \pa{M_2,M_3}`.
D'où :

    \begin{eqnarray}
    d\pa{m_1,m_3} \infegal d\pa{m_1,m_2} + d \pa{m_2,m_3} \label{edit_demo_eq_3}
    \end{eqnarray}

Les assertions :ref:`1 <edit_demo_eq_1>`, :ref:`2 <edit_demo_eq_2>`, :ref:`3 <edit_demo_eq_3>` 
montrent que :math:`d` est bien une distance. Le tableau suivant
illustre la démonstration pour les suites :math:`M_1,M_2,M_3` pour les mots
et les mots ``idtzance``, ``tonce``, ``distances``.


.. csv-table::
   :widths: 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
   
    :math:`M_1`, i, d,  , t, z, a, n, c, e,
    :math:`M_2`,  ,  ,  , t,  , o, n, c, e,
    :math:`M_3`, d, i, s, t,  , a, n, c, e, s




La distance d'édition :ref:`2 <defition_distance_edition_2>` 
ne tient pas compte de la longueur des mots qu'elle compare. On
serait tenté de définir une nouvelle distance d'édition inspirée de la précédente :

.. mathdef::
    :tag: Définition
    :title: distance d'édition étendue

    Soit `d^*` la distance d'édition définie en :ref:`2 <defition_distance_edition_2>` 
    pour laquelle les coûts de comparaison, d'insertion et de suppression 
    sont tous égaux à 1.    
    La distance d'édition :math:`d'` sur :math:`\mathcal{S}_\mathcal{C}` est définie par :
    
    .. math::
        :nowrap:
        :label: edit_equ_pseudo_dist
        
        \begin{eqnarray*}
        \begin{array}{crcl}
        d' : & \mathcal{S}_\mathcal{C} \times \mathcal{S}_\mathcal{C} & \longrightarrow & \R^+\\
        & \pa{m_1,m_2} & \longrightarrow & d'\pa{m_1,m_2} = \dfrac{d^*\pa{m_1,m_2}}{ \max \acc {l\pa{m_1}, l\pa{m_2}}} \\ \\
        & & & \text{où } l\pa{m} \text{ est la longueur du mot } m
        \end{array}
        \end{eqnarray*}

Le tableau suivant donne un exemple pour lequel l'inégalité triangulaire n'est pas
vérifiée. La fonction :math:`d^*` n'est donc pas une distance.

.. csv-table::
   :widths: 5, 5, 5, 5
   :header: mot 1, mot 2, distance : :math:`d^*`, distance :math:`d'`
   
    APPOLLINE, APPOLINE, 1, 1 / 9
    APPOLLINE, APOLLINE, 1, 1 / 9
    APOLLINE, APPOLINE, 2, 2 / 8

Par conséquent : :math:`d\pa{APOLLINE,APPOLINE} > d\pa{APOLLINE,APPOLLINE} + d\pa{APPOLLINE,APPOLINE}`
et la la fonction :math:`d^*` ne vérifie pas l'inégalité triangulaire.


Factorisation des calculs
=========================

La définition de la distance d'édition ne permet pas d'envisager le 
calcul de la distance dans un temps raisonnable. Il est possible néanmoins 
d'exprimer cette distance d'une autre manière afin de résoudre ce problème
(voir [Wagner1974]_). On définit la suite suivante :

.. mathdef::
    :tag: Définition
    :title: distance d'édition tronquée
    :label: definition_edit_dist_tronc
		
    Soient deux mots :math:`\pa{m_1,m_2}`, on définit la suite :
    
    .. math::
		    
        \left( d_{i,j}^{m_{1},m_{2}}\right) _{\substack{0\leqslant
        i\leqslant n_{1}\\0\leqslant j\leqslant n_{2}}}\left( =\left(d_{i,j}\right) _{\substack{0\leqslant i\leqslant
        n_{1}\\0\leqslant
        j\leqslant n_{2}}}\text{ pour ne pas alourdir les notations}\right)
		
    Par :
    
    .. math::
		
        \left\{
        \begin{array}[c]{l}%
        d_{0,0}=0\\
        d_{i,j}=\min\left\{
        \begin{array}{lll}
        d_{i-1,j-1}	&	+	& \text{comparaison}	\left(  m_1^i,m_2^j\right), \\
        d_{i,j-1}		&	+	& \text{insertion}		\left(  m_2^j\right), \\
        d_{i-1,j}		&	+	& \text{suppression}	\left(  m_1^i\right)
        \indexfr{comparaison}
        \indexfr{insertion}
        \indexfr{suppression}
        \end{array}
        \right\}%
        \end{array}
        \right.
		

Cette suite tronquée permet d'obtenir le résultat de la propriété suivante :

.. mathdef::
    :tag: Propriété
    :title: calcul rapide de la distance d'édition
    :label: edition_distance_propriete_001
    
    La suite définie en~\ref{definition_edit_dist_tronc} vérifie
    :math:`d\left(  m_{1},m_{2}\right)  =d_{n_{1},n_{2}}`
    où :math:`d` est la distance d'édition définie en :ref:`1 <defition_distance_edition_1>`_
    ou :ref:`2 <defition_distance_edition_2>`.


Cette factorisation des calculs est illustrée par les tableaux de 
cette :ref:`figure <figure_distance_edition_exemple_deux>`.
La démonstration s'effectue par récurrence, la définition :ref:`3 <definition_edit_dist_tronc>`  
est bien sûr équivalente :ref:`1 <defition_distance_edition_1>` 
pour des mots de longueur un. On suppose donc que ce résultat est 
vrai pour un couple de mots :math:`\pa{m_1,m_2}` de longueur :math:`\pa{l_1,l_2}` 
vérifiant :math:`l_1 \infegal i` et `l_2 \infegal j` avec au plus une égalité. 
Soit :math:`m` un mot, on note :math:`n` le nombre de lettres qu'il contient. 
On note  :math:`m\left(  l\right)` le mot formé des :math:`l` premières lettres de :math:`m`. 
Alors :

.. math::
    :nowrap:

    \begin{eqnarray*}
    d_{i,j}^{m_{1},m_{2}} &=& d\left(  m_{1}\left( i\right) ,m_{2}\left( j\right)  \right)\\
    d\left(  m_{1}\left(  i\right)  ,m_{2}\left( j\right) \right)  &=&
        \min\left\{
                \begin{array}{lll}%
                d\left(  m_{1}\left(  i-1\right)  ,m_{2}\left(  j-1\right)  \right)
                		&	+	& \text{comparaison}\left(  m_{1,i},m_{2,j}\right), \\
                d\left(  m_{1}\left(  i\right)  ,m_{2}\left(  j-1\right)  \right)
                		&	+	& \text{insertion}\left(  m_{2,j}\right), \\
                d\left(  m_{1}\left(  i-1\right)  ,m_{2}\left(  j\right)  \right) 
                		&	+	& \text{suppression}\left(  m_{1,i}\right)
                \end{array}
            \right\}
    \end{eqnarray*}

Le calcul factorisé de la distance d'édition entre deux mots de longueur 
:math:`l_1` et :math:`l_2` a un coût de l'ordre :math:`O\pa{l_1 l_2}`. 
Il est souvent illustré par un tableau comme celui de la figure suivante 
qui permet également de retrouver la meilleure séquence d'opérations permettant 
de passer du premier mot au second.


    .. math::
    
        \frame{$%
        \begin{array}{c}
            \begin{array}[c]{ccc}%
                \frame{$%
                \begin{array}[c]{cc}%
                    \searrow & \\
                    \text{dans ce sens,} \\
                    \text{c'est une } \\
                    \text{comparaison}%
                \end{array}
                $}
                &
                \frame{$%
                \begin{array}[c]{c}%
                    \longrightarrow j\\
                    \text{dans ce sens, c'est une insertion}%
                \end{array}
                $}
                &
                \\%
                \frame{$%
                \begin{array}[c]{ll}%
                    & \text{dans ce sens,}\\
                    \downarrow & \text{c'est une}\\
                    i & \text{suppression}%
                \end{array}
                $}
                &
                \frame{$%
                \begin{array}[c]{ccccccccccc}%
                    &  & d & i & s & t & a & n & c & e & s\\
                    & 0 &  &  &  &  &  &  &  &  & \\
                    i &  & 1 &  &  &  &  &  &  &  & \\
                    d &  &  & 2 &  &  &  &  &  &  & \\
                    s &  &  &  & 2 &  &  &  &  &  & \\
                    t &  &  &  &  & 2 &  &  &  &  & \\
                    z &  &  &  &  & 3 &  &  &  &  & \\
                    a &  &  &  &  &  & 3 &  &  &  & \\
                    n &  &  &  &  &  &  & 3 &  &  & \\
                    c &  &  &  &  &  &  &  & 3 &  & \\
                    e &  &  &  &  &  &  &  &  & 3 & 4
                \end{array}
                $}%
                &
                \frame{$
                \begin{array}
                [c]{ccccccccccc}%
                &  & d & i & s & t & a & n & c & e & s\\
                & 0 &  &  &  &  &  &  &  &  & \\
                i & 1 &  &  &  &  &  &  &  &  & \\
                d & 2 & 3 & 4 &  &  &  &  &  &  & \\
                s &  &  &  & 4 &  &  &  &  &  & \\
                t &  &  &  &  & 4 &  &  &  &  & \\
                z &  &  &  &  & 5 &  &  &  &  & \\
                a &  &  &  &  &  & 5 & 6 & 7 &  & \\
                n &  &  &  &  &  &  &  & 8 &  & \\
                c &  &  &  &  &  &  &  & 9 &  & \\
                e &  &  &  &  &  &  &  &  & 9 & 10
                \end{array}
                $}
            \end{array}
            \bigskip
            \\
            \begin{tabular}{c}%
                \begin{minipage}{15cm}
                Chaque case $\pa{i,j}$ contient la distance qui sépare les $i$ premières lettres du mot $1$
                des $j$ premières lettres du mot $2$ selon le chemin ou la méthode choisie.
                La dernière case indique la distance qui sépare les deux mots quel que soit le chemin choisi.
                \end{minipage}
            \end{tabular}
        \end{array}
        $}%
        


Extension de la distance d'édition
==================================

Jusqu'à présent, seuls trois types d'opérations ont été envisagés pour 
constuire la distance d'édition, tous trois portent sur des caractères et 
aucunement sur des paires de caractères. L'article [Kripasundar1996]_ 
(voir aussi [Seni1996]_ suggère d'étendre la définition :ref:`3 <definition_edit_dist_tronc>` 
aux permutations de lettres :


.. mathdef::
    :tag: Définition
    :title: distance d'édition tronquée étendue
    :label: definition_edit_dist_tronc_2
		
    Soit deux mots :math:`\pa{m_1,m_2}`, on définit la suite :
    
    .. math::
    
        \left( d_{i,j}^{m_{1},m_{2}}\right) _{\substack{0\leqslant
        i\leqslant n_{1}\\0\leqslant j\leqslant n_{2}}}\left( =\left(d_{i,j}\right) _{\substack{0\leqslant i\leqslant
        n_{1}\\0\leqslant
        j\leqslant n_{2}}}\text{ pour ne pas alourdir les notations}\right)

    par :
    
    .. math::
		
        \left\{
        \begin{array}[c]{l}%
        d_{0,0}=0\\
        d_{i,j}=\min\left\{
        \begin{array}{lll}
        d_{i-1,j-1} & + &   \text{comparaison}  \pa{m_1^i,m_2^j},      \\
        d_{i,j-1}   & + &   \text{insertion}    \pa{m_2^j,i},          \\
        d_{i-1,j}   & + &   \text{suppression}  \pa{m_1^i,j},          \\
        d_{i-2,j-2} & + &   \text{permutation}  \pa{ \pa{m_1^{i-1}, m_1^i},\pa{m_2^{j-1}, m_2^j}}
        \end{array}
        \right\}%
        \end{array}
        \right.

La distance d'édition cherchée est toujours :math:`d\pa{m_1,m_2} = d_{n_1,n_2}` 
mais la démonstration du fait que :math:`d` est bien une distance ne peut pas 
être copiée sur celle du théorème :ref:`1 <edition_distance_theoreme001>` 
mais sur les travaux présentés dans l'article [Wagner1974]_.



Apprentissage d'une distance d'édition
======================================

L'article [Waard1995]_ suggère l'apprentissage des coûts des opérations 
élémentaires associées à une distance d'édition (comparaison, insertion, 
suppression, permutation,~...). On note l'ensemble de ces coûts ou 
paramètres :math:`\Theta = \vecteur{\theta_1}{\theta_n}`. 
On considère deux mots :math:`X` et :math:`Y`, la distance d'édition :math:`d\pa{X,Y}` 
est une fonction linéaire des coûts. Soit :math:`D = \vecteur{\pa{X_1,Y_1}}{\pa{X_N,Y_N}}` 
une liste de couple de mots pour lesquels le résultat de la distance 
d'édition est connu et noté :math:`\vecteur{c_1}{c_N}`, il est alors 
possible de calculer une erreur s'exprimant sous la forme : 

.. math::
    :nowrap:

    \begin{eqnarray*}
    E = \sum_{i=1}^{N} \; \pa{d\pa{X_i,Y_i} - c_i}^2 =\sum_{i=1}^{N} \; 
                \pa{ \sum_{k=1}^{n} \alpha_{ik}\pa{\Theta} \, \theta_k - c_i}^2 \\
    \end{eqnarray*}			

Les coefficients :math:`\alpha_{ik}\pa{\Theta}` dépendent des paramètres :math:`\Theta` 
car la distance d'édition correspond au coût de la transformation de moindre coût 
d'après la définition :ref`2 <defition_distance_edition_2>`, 
:math:`\alpha_{ik}\pa{\Theta}` correspond au nombre de fois que le paramètre 
:math:`\theta_k` intervient dans la transformation de moindre coût entre 
:math:`X_i` et :math:`Y_i`. Cette expression doit être minimale afin d'optenir 
les coûts :math:`\Theta` optimaux. Toutefois, les coûts :math:`\theta_k` sont tous 
strictement positifs et plutôt que d'effectuer une optimisation sous 
contrainte, ces coûts sont modélisés de la façon suivante :

.. math::
    :nowrap:
    :label: edit_distance_eq_2_app
    
    \begin{eqnarray*}
    E = \sum_{i=1}^{N} \; \pa{ \sum_{k=1}^{n} \, \alpha_{ik}\pa{\Omega} \, \frac{1}{1 + e^{-\omega_k}} - c_i}^2
    \end{eqnarray*}			


Les fonctions :math:`\alpha_{ik}\pa{\Omega}` ne sont pas dérivable par rapport 
:math:`\Omega` mais il est possible d'effectuer une optimisation sans contrainte 
par descente de gradient. Les coûts sont donc appris en deux étapes :


.. mathdef::
    :tag: Algorithme
    :title: Apprentissage d'une distance d'édition
    :lid: edit_distance_app_optom
    
    Les notations sont celles utilisés pour l'équation :eq:`edit_distance_eq_2_app`. 
    Les coûts :math:`\Omega` sont tirés aléatoirement.
    
    *estimation* 
    
    Les coefficients :math:`\alpha_{ik}\pa{\Omega}` sont calculées.
    
    *calcul du gradient*
    
    Dans cette étape, les coefficients :math:`\alpha_{ik}\pa{\Omega}` 
    restent constants. Il suffit alors de minimiser la fonction
    dérivable :math:`E\pa{\Omega}` sur :math:`\R^n`, ceci peut être 
    effectué au moyen d'un algorithme de descente de gradient
    similaire à ceux utilisés pour les réseaux de neurones.
    
    Tant que l'erreur :math:`E\pa{\Omega}` ne converge pas, on continue.
    L'erreur `E` diminue jusqu'à converger puisque l'étape qui réestime les coefficients 
    :math:`\alpha_{ik}\pa{\Omega}`, les minimise à :math:`\Omega = \vecteur{\omega_1}{\omega_n}` constant.


Bibliographie
=============

.. [Damerau1964] A technique for computer detection and correction of spelling errors (1964),
    *F. J. Damerau*, Commun. ACM, volume 7(3), pages 171-176

.. [Kripasundar1996] Generating edit distance to incorporate domain information (1996),
    *V. Kripasunder, G. Seni, R. K. Srihari*, CEDAR/SUNY

.. [Levenstein1966] Binary codes capables of correctiong deletions, insertions, and reversals (1966),
    *V. I. Levenstein*, Soviet Physics Doklady, volume 10(8), pages 707-710

.. [Seni1996] Generalizing edit distance to incorporate domain information: handwritten text recognition as a case study (1996),
    *Giovanni Seni, V. Kripasundar, Rohini K. Srihari*, Pattern Recognition volume 29, pages 405-414

.. [Waard1995] An optimised minimal edit distance for hand-written word recognition (1995),
    *W. P. de Waard*, Pattern Recognition Letters volume 1995, pages 1091-1096

.. [Wagner1974] The string-to-string correction problem (1974),
    *R. A. Wagner, M. Fisher*, Journal of the ACM, volume 21, pages 168-178


