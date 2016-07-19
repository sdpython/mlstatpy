
.. _nn-acp:

Analyse en composantes principales (ACP) et Auto Encoders
=========================================================

.. index:: ACP


Cet algorithme est proposé dans [Song1997]_.
Autrefois réseau diabolo, le terme `auto-encoder <https://en.wikipedia.org/wiki/Autoencoder>`_
est plus utilisé depuis l'avénement du deep learning. Il s'agit de compresser avec perte 
un ensemble de points. L'`ACP <https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales>`_ 
est une forme de compression linéaire puisqu'on cherche 
à préserver l'information en projetant un nuage de points de façon à maximiser
l'inertie du nuage. Les auto-encoders fonctionnent sur le même principe
avec des modèles non linéaires.

.. index: diabolo

\subsection{Principe}


L'algorithme implémentant l'analyse en composantes principales 
est basé sur un réseau linéaire dit "diabolo", ce réseau
possède une couche d'entrées à :math:`N` entrées, une couche cachée et une couche 
de sortie à :math:`N` sorties. L'objectif est
d'apprendre la fonction identité sur l'espace :math:`\R^N`. 
Ce ne sont plus les sorties qui nous intéressent mais la couche
cachée intermédiaire qui effectue une compression ou projection 
des vecteurs d'entrées puisque les entrées et les
sorties du réseau auront pour but d'être identiques. 


.. mathdef:: 
    :title: Principe de la compression par un réseau diabolo
    :tag: Figure
    :lid: figure_rn_acp-fig

    .. math::
        :nowrap:

        \begin{picture}(241,100)(0,-10)

        \put(1,1)   {\framebox(40,22){\footnotesize \begin{tabular}{c}vecteur \\ $X \in \R^N$ \end{tabular}}}
        \put(85,-9)  {\framebox(45,32){\footnotesize \begin{tabular}{c}vecteur \\ $Y \in \R^M$ \\ et $M < N$ \end{tabular}}}
        \put(200,1) {\framebox(40,22){\footnotesize \begin{tabular}{c}vecteur \\ $Z \approx X$ \end{tabular}}}

        \put(20,40) {\framebox(90,45){\footnotesize
                                        \begin{minipage}{30mm} première couche du réseau diabolo~:
                                        \textbf{projection (ou compression)}
                                        \end{minipage}}}

        \put(120,40) {\framebox(90,45){\footnotesize
                                        \begin{minipage}{30mm} seconde couche du réseau diabolo~:
                                        \textbf{reconstitution (ou décompression)}
                                        \end{minipage}}}
        \put(30,23) {\vector(1,1){17}}
        \put(130,23) {\vector(1,1){17}}

        \put(90,39) {\vector(1,-1){17}}
        \put(190,39) {\vector(1,-1){17}}


        \end{picture}


La figure suivante illustre un exemple de compression de vecteur de :math:`\R^3` 
dans :math:`\R^2`.

.. mathdef:: 
    :title: Réseau diabolo : réduction d'une dimension
    :tag: Figure
    :lid: figure_rn_acp-exemple

    .. math::
        :nowrap:
        
        \begin{picture}(130,75)(0,0)

        \put(20,10) {\circle{20}}
        \put(20,40) {\circle{20}}
        \put(20,70) {\circle{20}}

        \put(18,8) {\makebox(5,5){\footnotesize $x_1$}}
        \put(18,38) {\makebox(5,5){\footnotesize $x_2$}}
        \put(18,68) {\makebox(5,5){\footnotesize $x_3$}}


        \put(65,25) {\circle{20}}
        \put(65,55) {\circle{20}}

        \put(63,23) {\makebox(5,5){\footnotesize $z_{1,1}$}}
        \put(63,53) {\makebox(5,5){\footnotesize $z_{1,2}$}}


        \put(110,10) {\circle{20}}
        \put(110,40) {\circle{20}}
        \put(110,70) {\circle{20}}

        \put(108,8) {\makebox(5,5){\footnotesize $z_{2,1}$}}
        \put(108,38) {\makebox(5,5){\footnotesize $z_{2,2}$}}
        \put(108,68) {\makebox(5,5){\footnotesize $z_{2,3}$}}

        \drawline(30,10)(55,25)
        \drawline(30,40)(55,55)
        \drawline(30,10)(55,55)

        \drawline(30,70)(55,25)
        \drawline(30,70)(55,55)
        \drawline(30,40)(55,25)

        \drawline(75,25)(100,10)
        \drawline(75,25)(100,40)
        \drawline(75,25)(100,70)

        \drawline(75,55)(100,10)
        \drawline(75,55)(100,40)
        \drawline(75,55)(100,70)

        \end{picture}        

    Ce réseau possède 3 entrées et 3 sorties
    Minimiser l'erreur :math:`\sum_{k=1}^N E\left(  X_{k},X_{k}\right)`
    revient à compresser un vecteur de dimension 3 en un vecteur de dimension 2. 
    Les coefficients de la
    première couche du réseau de neurones permettent de compresser les données. 
    Les coefficients de la seconde couche permettent de les décompresser.



La compression et décompression ne sont pas inverses 
l'une de l'autre, à moins que l'erreur :eq:`rn_equation_acp_error` soit nulle. 
La décompression s'effectue donc avec des pertes d'information. 
L'enjeu de l'ACP est de trouver un bon compromis entre le nombre 
de coefficients et la perte d'information tôlérée. 
Dans le cas de l'ACP, la compression est "linéaire", c'est une projection.





.. _par_ACP_un:


Problème de l'analyse en composantes principales
++++++++++++++++++++++++++++++++++++++++++++++++



L'analyse en composantes principales ou ACP est définie de la manière suivante :

.. mathdef::
    :title: analyse en composantes principales (ACP)
    :lid: problem_acp
    :tag: Problème

    Soit :math:`\pa{X_i}_{1 \infegal i \infegal N}` avec :math:`\forall i \in \ensemble{1}{N}, 
    \; X_i \in \R^p`.
    Soit :math:`W \in M_{p,d}\pa{\R}`, :math:`W = \vecteur{C_1}{C_d}`
    où les vecteurs :math:`\pa{C_i}` 
    sont les colonnes de :math:`W` et :math:`d < p`.
    On suppose également que les :math:`\pa{C_i}` forment une base othonormée.
    Par conséquent :
    
    .. math::
    
        W'W = I_d
    
    :math:`\pa{W'X_i}_{1 \infegal i \infegal N}` est l'ensemble des 
    vecteurs :math:`\pa{X_i}` projetés sur le sous-espace vectoriel
    engendré par les vecteurs :math:`\pa{C_i}`.
    Réaliser une analyse en composantes principales, c'est trouver le 
    meilleur plan de projection pour les vecteurs
    :math:`\pa{X_i}`, celui qui maximise l'inertie de ce nuage de points, 
    c'est donc trouver :math:`W^*` tel que :
    
    .. math::
        :nowrap:
        :label: rn_equation_acp_error
    
        \begin{eqnarray*}
        W^* &=& \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } 
                                            { \arg \max } \; E\pa{W}
            =  \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \arg \max } \;
                            \cro { \sum_{i=1}^{N} \norm{W'X_i}^2 } 
        \end{eqnarray*}
    
    Le terme :math:`E\pa{W}` est l'inertie du nuage de points :math:`\pa{X_i}` 
    projeté sur le sous-espace vectoriel défini par les
    vecteurs colonnes de la matrice :math:`W`.
    
		


Résolution d'une ACP avec un réseau de neurones diabolo
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Un théorème est nécessaire avant de construire le réseau de 
neurones menant à la résolution du problème de l':ref:`ACP <problem_acp>` 
afin de passer d'une optimisation sous contrainte à une optimisation sans contrainte. 


.. mathdef::
    :title: résolution de l'ACP
    :lid: theorem_acp_resolution
    :tag: Théorème

    Les notations utilisées sont celles du problème de l':ref:`ACP <problem_acp>`. 
    Dans ce cas :
    
    .. math::
        :nowrap:
        :label: rn_acp_contrainte
		
        \begin{eqnarray*}
        S =
        \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \arg \max } \;
                            \cro { \sum_{i=1}^{N} \norm{W'X_i}^2 } &=&
        \underset{ W \in M_{p,d}\pa{\R} } { \arg \min } \;  \cro { \sum_{i=1}^{N} \norm{WW'X_i - X_i}^2 }
        \end{eqnarray*}
		
    De plus :math:`S` est l'espace vectoriel engendré par les :math:`d`
    vecteurs propres de la matrice 
    :math:`XX' = \sum_{i=1}^{N} X_i X_i'` associées aux
    :math:`d` valeurs propres de plus grand module. 


**Démonstration**

*Partie 1*


L'objectif de cette partie est de chercher la valeur de :

.. math::

    \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \max }\; E\pa{W}

Soit :math:`X=\vecteur{X_1}{X_N} \in \pa{\R^p}^N`, alors :

.. math:: 

    E\pa{W} = \sum_{i=1}^{N} \norm{W'X_i}^2 = \trace{X'WW'X} = \trace{XX'WW'}
    

La matrice :math:`XX'` est symétrique, elle est donc diagonalisable 
et il existe une matrice :math:`P \in M_p\pa{\R}:math:` telle qu :

.. math::
    :label: acp_equation_memo_1

    \begin{array}{l}
    P'XX'P = D_X \text{ avec } D_X \text{ diagonale} \\
    P'P = I_p
    \end{array}

Soit :math:`P = \vecteur{P_1}{P_p}` les vecteurs propres de la matrice 
:math:`XX'` associés aux valeurs propres
:math:`\vecteur{\lambda_1}{\lambda_p}` telles que 
:math:`\abs{\lambda_1} \supegal ... \supegal \abs{\lambda_p}`. 
Pour mémoire, :math:`W = \vecteur{C_1}{C_d}`, et on a :

.. math::

    \begin{array}{l}
    \forall i \in \ensemble{1}{p}, \; XX'P_i = \lambda_i P_i \\
    \forall i \in \ensemble{1}{d}, \; C_i = P_i \Longrightarrow XX'WW' = D_{X,d} = \pa{
                                                        \begin{array}{ccc}
                                                        \lambda_1 & 0 & 0 \\
                                                        0  & \ldots & 0 \\
                                                        0 & 0 & \lambda_d
                                                        \end{array}
                                                        }
    \end{array}

D'où :

.. math::

    E\pa{W} = \trace{ XX'WW' } = \trace{P D_X P' WW'} = \trace{ D_X P'WW'P }

Donc :

.. math::
    :nowrap:
    :label: acp_demo_partie_a

    \begin{eqnarray*}
    \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \max }\; E\pa{W} =
            \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \max }\; 
            	\trace{ D_X P'WW'P }
    = \underset{ \begin{subarray}{c} Y \in M_{p,d}\pa{\R} \\ Y'Y = I_d \end{subarray} } { \max }\; \trace{ D_X YY'
                }
    = \sum_{i=1}{d} \lambda_i
    \end{eqnarray*}


*Partie 2*


Soit :math:`Y \in \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \max }\; \trace{X'WW'X}`, 
:math:`Y = \vecteur{Y_1}{Y_d} = \pa{y_i^k}_{ \begin{subarray}{c} 1 \infegal i \infegal d \\ 1 \infegal k \infegal p \end{subarray} }`.

Chaque vecteur :math:`Y_i` est écrit dans la base 
:math:`\vecteur{P_1}{P_p}` définie en :eq:`acp_equation_memo_1` :

.. math::

    \forall i \in \ensemble{1}{d}, \; Y_i = \sum_{k=1}^{p} y_i^k P_p

Comme :math:`Y'Y = I_d`, les vecteurs :math:`\vecteur{Y_1}{Y_d}` 
sont orthogonaux deux à deux et normés, ils vérifient donc :

.. math::

    \left\{
    \begin{array}{rl}
    \forall i \in \ensemble{1}{d},          & \sum_{k=1}^{p} \pa{y_i^k}^2 = 1 \\
    \forall \pa{i,j} \in \ensemble{1}{d}^2, & \sum_{k=1}^{p} y_i^k y_j^k = 0
    \end{array}
    \right.


De plus :

.. math::

    XX'YY' = XX' \pa{ \sum_{i=1}^{d} Y_i Y_i'} =   \sum_{i=1}^{d} XX' Y_i Y_i'

On en déduit que :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \forall i \in \ensemble{1}{d}, \; XX' Y_i Y'_i
                &=& XX' \pa{ \sum_{k=1}^{p} y_i^k P_k }\pa{ \sum_{k=1}^{p} y_i^k P_k }' \\
                &=& \pa{ \sum_{k=1}^{p} \lambda_k y_i^k P_k }\pa{ \sum_{k=1}^{p} y_i^k P_k }'
    \end{eqnarray*}

D'où :

.. math::

    \forall i \in \ensemble{1}{d}, \; \trace{ XX' Y_i Y'_i} = \sum_{k=1}^{p} \lambda_k \pa{y_i^k}^2

Et :

.. math::
    :nowrap:

    \begin{eqnarray*}
    \trace{ XX' YY'} &=& \sum_{i=1}^{d} \sum_{k=1}^{p} \lambda_k \pa{y_i^k}^2 \\
    \trace{ XX' YY'} &=& \sum_{k=1}^{p} \lambda_k \pa {\sum_{i=1}^{d} \pa{y_i^k}^2} =
    				\sum_{k=1}^{p} \; \lambda_k
    \end{eqnarray*}

Ceci permet d'affirmer que :

.. math::
    :nowrap:
    :label: acp_demo_partie_b

    \begin{eqnarray*}
    Y \in \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \max }\;
                \trace{X'WW'X}  \Longrightarrow
    vect \vecteur{Y_1}{Y_d} = vect \vecteur{P_1}{P_d}
    \end{eqnarray*}

Les équations :eq:`acp_demo_partie_a` et :eq:`acp_demo_partie_b` démontrent la seconde partie du
théorème.


*Partie 3*

.. math::
    :nowrap:

    \begin{eqnarray*}
    \sum_{i=1}^n \left\|  WW^{\prime}X_{i}-X_{i}\right\|^{2} &=&
    \sum_{i=1}^n \left\|
        \left(  WW^{\prime} -I_{N}\right)  X_{i}\right\|  ^{2} \\
    &=& tr\left(  X^{\prime}\left(  WW^{\prime }-I_{p}\right)  ^{2}X\right)  \\
    &=& tr\left(  XX^{\prime}\left(  \left( WW^{\prime}\right) ^{2}-2WW^{\prime}+I_{p}\right)  \right) \\
    &=& tr\left(  XX^{\prime}\left(  WW^{\prime}WW^{\prime}-2WW^{\prime}+I_{p}\right)  \right) \\
    &=& tr\left(  XX^{\prime}\left(  -WW^{\prime} +I_{p}\right)  \right) \\
    &=& -tr\left(  XX^{\prime}WW^{\prime}\right)  +tr\left(XX^{\prime}\right)
    \end{eqnarray*}

D'où :

.. math::
    :nowrap:
    :label: acp_demo_partie_c

    \begin{eqnarray*}
    \underset{ \begin{subarray} \, W \in M_{p,d} \pa{\R} \\ 
    						W'W=I_d \end{subarray}} { \; \max \; } \;  \pa {  \sum_{i=1}^{N} \norm{ W'X_i}^2 }  =
    \underset{ \begin{subarray} \, W \in M_{p,d} \pa{\R} \\ 
    						W'W=I_d \end{subarray}} { \; \min \; } \;  \pa {  \sum_{i=1}^{N} \norm{ WW'X_i - X_i}^2 }
    \end{eqnarray*}


*Partie 4*

:math:`XX'` est une matrice symétrique, elle est donc diagonalisable :

.. math:: 

    \exists P\in GL_N \pa{\R}  \text{ telle que } P'XX'P=D_p \text{ où } D_p \text{ est diagonale}

On en déduit que :

.. math::
    :nowrap:

    \begin{eqnarray*}
        \sum_{i=1}^{N} \norm{  WW' X_i - X_i }^2
    &=& \trace{ XX' \pa{ WW'-I_p }^{2} } \\
    &=& \trace{ PP' XX' PP' \pa{ WW'-I_p }^{2} } \\
    &=& \trace{ P D_p P' \pa{ WW'-I_p }^{2} } \\
    &=& \trace{ D_p \pa{ P'WW'P-I_p }^{2} } \\
    &=& \trace{ D_p \pa{ YY'-I_p }^{2} } \text{ avec } Y = P'W
    \end{eqnarray*}

D'où :

.. math::
    :nowrap:
    :label: acp_demo_partie_d

    \begin{eqnarray*}
    \underset{Y}{\arg\min}\acc{ tr\left(  D_{p}\left( YY^{\prime}-I_{p}\right)  ^{2}\right)}  = \left\{  Y\in
    M_{Nd}\left( \R\right) \left|
        YY^{\prime}=I_{d}\right.  \right\}
    \end{eqnarray*}


Finalement, l'équation :eq:`acp_demo_partie_d` permet de démontrer la 
première partie du théorème, à savoir :eq:`rn_acp_contrainte` :

.. math::
    :nowrap:

    \begin{eqnarray*}
    S =
    \underset{ \begin{subarray}{c} W \in M_{p,d}\pa{\R} \\ W'W = I_d \end{subarray} } { \arg \max } \;
                        \cro { \sum_{i=1}^{N} \norm{W'X_i}^2 } &=&
    \underset{ W \in M_{p,d}\pa{\R} } { \arg \min } \;  \cro { \sum_{i=1}^{N} \norm{WW'X_i - X_i}^2 }
    \end{eqnarray*}


.. _par_ACP_deux:

Calcul de valeurs propres et de vecteurs propres
++++++++++++++++++++++++++++++++++++++++++++++++


Le calcul des valeurs propres et des vecteurs propres d'une 
matrice fait intervenir un réseau diabolo composé d'une
seule couche cachée et d'une couche de sortie avec des fonctions 
de transfert linéaires. On note sous forme de matrice
:math:`\left( W\right)` les coefficients de la seconde couche 
du réseau dont les biais sont nuls. On note :math:`d` le nombre de
neurones sur la couche cachée, et :math:`p` le nombre d'entrées.

.. math::

    \forall i\in\left\{  1,...,d\right\}  ,\,y_{1,i}=\sum_{j=1}^p w_{ji}x_{j}
    
Soit :math:`X\in\R^{p}` les entrées, 
:math:`Y=\left(  y_{1,1},...,y_{1,d}\right)  \in\R^{d}`, 
on obtient que : :math:`Y=W'X`.

Les poids de la seconde couche sont définis comme suit :

.. math:: 

    \forall\left( i,j\right)  \in\left\{  1,...,p\right\}  \times\left\{ 1,...,d\right\} \,w_{2,j,i}=w_{1,i,j}

Par conséquent, le vecteur des sorties :math:`Z\in\R^{p}` 
du réseau ainsi construit est :math:`Z=WW'X`.
On veut minimiser l'erreur pour :math:`\left(  X_{i}\right)  _{1\leqslant i\leqslant N}` :

.. math::

    E=\sum_{i=1}^N\left\|  WW'X_{i}-X_{i}\right\|  ^{2}

Il suffit d'apprendre le réseau de neurones pour obtenir :

.. math::

    W_{d}^{\ast}=\underset{W\in M_{pd}\left(  \R\right)  }
    {\arg\max }\,\sum_{i=1}^N\left\| WW'X_{i}-X_{i}\right\|
    ^{2}

D'après ce qui précède, l'espace engendré par les vecteurs 
colonnes de :math:`W` est l'espace engendré par les :math:`k` 
premiers vecteurs propres de la matrice 
:math:`XX^{\prime}=\left(  X_{1},...,X_{P}\right)  \left( X_{1},...,X_{P}\right)  ^{\prime}` 
associés aux :math:`k` premières valeurs propres classées par ordre décroissant de module.


On en déduit que :math:`W_{1}^{\ast}` est le vecteur propre de la matrice 
:math:`M` associée à la valeur propre de plus grand module. 
:math:`W_{2}^{\ast}` est l'espace engendré par les deux premiers vecteurs. 
Grâce à une `orthonormalisation de Schmidt <https://fr.wikipedia.org/wiki/Algorithme_de_Gram-Schmidt>`_.
On en déduit à partir de :math:`W_{1}^{\ast}` et :math:`W_{2}^{\ast}`, 
les deux premiers vecteurs propres. Par récurrence, 
on trouve l'ensemble des vecteurs propres de la matrice :math:`XX^{\prime}`.

.. mathdef::
    :title: orthonormalisation de Schmidt
    :tag: Définition
    :lid: orthonormalisation_schmidt

    L'orthonormalisation de Shmidt :
    
    Soit :math:`\left(  e_{i}\right)  _{1\leqslant i\leqslant N}` 
    une base de :math:`\R^{p}`
    
    On définit la famille :math:`\left(  \varepsilon_{i}\right)  _{1\leqslant i\leqslant p}` 
    par :
    
    .. math:: 
        :nowrap:
        
        \begin{eqnarray*}
        \varepsilon_{1} &=& \dfrac{e_{1}}{\left\| e_{1}\right\|}\\
        \forall i \in \intervalle{1}{p}, \; \varepsilon_{i} &=& \dfrac{e_{i}-\overset{i-1}{\underset{j=1}
        {\sum}}<e_{i},\varepsilon_{j}>\varepsilon_{j}}{\left\| 
                    e_{i}-\overset {i-1}{\underset{j=1}{\sum}}<e_{i},\varepsilon_{j}>\varepsilon_{j}\right\| }
        \end{eqnarray*}
    
    
On vérifie que le dénominateur n'est jamais nul.
:math:`e_{i}-\overset{i-1}{\underset{j=1}{\sum}}<e_{i},\varepsilon_{j}>\varepsilon_{j}\neq 0` 
car :math:`\forall k\in\left\{ 1,...,N\right\}  ,\; vect\left( e_{1},...,e_{k}\right)  
=vect\left(  \varepsilon_{1} ,...,\varepsilon_{k}\right)`


.. mathdef::
    :title: base orthonormée
    :tag: Propriété

    La famille :math:`\left(  \varepsilon_{i}\right)  _{1\leqslant i\leqslant p}` 
    est une base orthonormée de :math:`\R^{p}`.


L'algorithme qui permet de déterminer les vecteurs propres de la matrice :math:`XX'` 
définie par le théorème de l':ref:`ACP <theorem_acp_resolution>` est le suivant :

.. mathdef::
    :title: vecteurs propres
    :lid: algorithm_vecteur_propre
    :tag: Algorithme

    Les notations utilisées sont celles du théorème de l':ref:`ACP <theorem_acp_resolution>`. 
    On note :math:`V^*_d` la matrice des :math:`d`
    vecteurs propres de la matrice :math:`XX'` associés aux 
    :math:`d` valeurs propres de plus grands module.
    
    | for :math:`d, p`
    |   Un réseau diabolo est construit avec les poids :math:`W_d \in M_{p,d}\pa{\R}` puis appris. 
    |   Le résultat de cet apprentissage sont les poids :math:`W^*_d`.
    |   if :math:`d > 1`
    |       L'orthonormalisation de Schmit permet de déduire :math:`V^*_d` de :math:`V^*_{d-1}` et :math:`W^*_d`.
    |   else
    |       :math:`V^*_d = W^*_d`



Analyse en Composantes Principales (ACP)
++++++++++++++++++++++++++++++++++++++++


L'analyse en composantes principales permet d'analyser 
une liste d'individus décrits par des variables. 
Comme exemple, il suffit de prendre les informations 
extraites du recensement de la population française 
qui permet de décrire chaque habitant par des 
variables telles que la catégorie socio-professionnelle, 
la salaire ou le niveau d'étude.
Soit :math:`\left(  X_{1},...,X_{N}\right)` un ensemble de 
:math:`N` individus décrits par :math:`p` variables :

.. math:: 

    \forall i\in\left\{  1,...,N\right\},\;X_{i}\in\R^{p}
    
L'ACP consiste à projeter ce nuage de point sur un plan 
qui conserve le maximum d'information. Par conséquent, il
s'agit de résoudre le problème :

.. math::

    W^{\ast}=\underset{ \begin{subarray} \, W\in M_{p,d}\left(  \R\right)  \\ 
    W^{\prime }W=I_{d} \end{subarray}}{\arg\min}%
    \left(\underset{i=1}{\overset{N}{\sum}}\left\| W'X_{i}\right\|  ^{2}\right)  \text{ avec }d<N

Ce problème a été résolu dans les paragraphes :ref:`par_ACP_un` 
et :ref:`par_ACP_deux`, il suffit d'appliquer
l'algorithme :ref:`vecteurs propres <algorithm_vecteur_propre>`.



Soit :math:`\left(  X_{i}\right)  _{1\leqslant i\leqslant N}` avec 
:math:`\forall i\in\left\{  1,...,N\right\} ,\,X_{i}\in\R^{p}`. 
Soit :math:`\pa{P_1,\dots,P_p}` l'ensemble des vecteurs propres 
normés de la matrice :math:`XX'` associés aux valeurs propres 
:math:`\pa{\lambda_1,\dots,\lambda_p}` classées par ordre décroissant de modules. 
On définit :math:`\forall d \in \intervalle{1}{p}, \; W_d = \pa{P_1,\dots,P_d} \in M_{p,d}`. 
On définit alors l'inertie :math:`I_d` du nuage de points projeté sur 
l'espace vectoriel défini par :math:`P_d`.
On suppose que le nuage de points est centré, alors :

.. math::

		\forall d \in \intervalle{1}{p}, \; I_d = \sum_{k=1}^{N} 
		\pa{P_d' X_k}^2 = tr \pa{X' P_d P_d' X} = tr \pa{XX' P_d P_d'} = \lambda_d

Comme :math:`\pa{P_1,\dots,P_p}` est une base orthonormée de :math:`\R^p`, 
on en déduit que :

.. math:: 

    I = \sum_{k=1}^{P} X_k'X_k = \sum_{d=1}^{N} I_d = \sum_{d=1}^{p} \lambda_d

De manière empirique, on observe fréquemment que la courbe 
:math:`\pa{d,I_d}_{1 \infegal d \infegal p}` montre un point
d'inflexion (voir figure ci-dessous). Dans cet exemple, le point 
d'inflexion correspond à :math:`d=4`. En
analyse des données, on considère empiriquement que seuls les 
quatres premières dimensions contiennent de l'information.

.. mathdef::
    :title: Courbe d'inertie pour l'ACP
    :tag: Figure
    :lid: figure_point_inflexion
    
    .. image:: rnimg/acp_inertie.png

    Courbe d'inertie : point d'inflexion pour :math:`d=4`, 
    l'expérience montre que généralement, seules les
    projections sur un ou plusieurs des quatre premiers vecteurs propres 
    reflètera l'information contenue par le nuage de points.

