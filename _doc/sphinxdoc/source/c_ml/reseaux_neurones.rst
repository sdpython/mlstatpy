

===================
Réseaux de neurones
===================


Ce chapitre aborde les réseaux de neurones au travers de deux utilisations courantes, 
la :ref:`régression <nn-regression>`
et la :ref:`classification <nn-classification>`
et une qui l'est moins, 
l'`analyse en composantes principales <https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales>`_
ou :ref:`ACP <nn-ACP>`
sans oublier les méthodes d'estimation des paramètres qui les composent, 
à savoir optimisations du premier et second ordre 
(:ref:`nn-rn_optim_premier_ordre`) et :ref`nn-rn_optim_second_ordre`
ainsi qu'une méthode permettant de supprimer des coefficients inutiles 
:ref`nn-selection_connexion`.

.. contents:: .
    :depth: 2


Définition des réseaux de neurones multi-couches
================================================


Les réseaux de neurones multi-couches (ou perceptrons) définissent une 
classe de fonctions dont l'intérêt est de pouvoir approcher n'importe quelle 
fonction continue à support compact 
(voir théorème :ref:`nn-theoreme_densite`).
Aucun autre type de réseau de neurones ne sera étudié et par la suite, 
tout réseau de neurones sera considéré comme multi-couches
(donc pas les `réseau de Kohonen <https://fr.wikipedia.org/wiki/Carte_auto_adaptative>`_).


Un neurone
++++++++++


.. mathdef:: 
    :title: neurone
    :tag: Définition
    :lid: def-neurone
    
    Un neurone à :math:`p` entrées est une fonction
    :math:`f : \R^{p+1} \times \R^p \longrightarrow \R`
    définie par :
    
    * :math:`g : \R \longrightarrow \R`
    * :math:`W \in \R^{p+1}`, :math:`W=\pa{w_1,\dots,w_{p+1}}`
    * :math:`\forall x \in \R^p, \; f\pa{W,x} = g \pa { \sum_{i=1}^{p} w_i x_i + w_{p+1}}`        
      avec :math:`x = \pa{x_1,\dots,x_p}`

Cette définition est inspirée du neurone biologique, les poids jouant le rôle 
de synapses, le vecteur :math:`x` celui des *entrées*
et :math:`W` celui des *coefficients* ou *poids*. 
Le coefficient :math:`w_{p+1}` est appelé le *biais* et souvent noté :math:`b`. 
La fonction *g* est appelée *fonction de transfert* ou *fonction de seuil*. 


.. mathdef:: 
    :title: neurone graphique
    :tag: Figure
    :lid: fig-nn-neurone

    .. math::
        :nowrap:

        \begin{picture}(100,80)(0,0)
        \put(10,0)  {\circle{20}}
        \put(10,25) {\circle{20}}
        \put(10,50) {\circle{20}}

        \put(10,0)  {\makebox(3,3){$x_1$}}
        \put(10,25) {\makebox(3,3){$x_i$}}
        \put(10,50) {\makebox(3,3){$x_p$}}

        \put(80,25) {\circle{35}}
        \put(78,25) {\makebox(6,3){$\;y \overset{f}{\rightarrow} z$}}

        \put(20,25) {\line(1,0){43}}
        \drawline(20,0)(63,25)
        \drawline(20,50)(63,25)

        \put(30,50)  {\makebox(3,3){$w_p$}}
        \put(30,18)  {\makebox(3,3){$w_i$}}
        \put(30,-2)  {\makebox(3,3){$w_1$}}

        \put(48,20)  {\makebox(3,3){$\sum$}}

        \put(50,-20)  {\circle{20}}
        \put(50,-20)  {\makebox(3,3){$1$}}
        \drawline(50,-10)(63,25)
        \put(50,5)  {\makebox(3,3){$b$}}

        \end{picture}

    Le vecteur :math:`\left(  x_1,...,x_p\right) \in \R^p`         
    joue le rôle des *entrées*.
    :math:`y` est appelé parfois le *potentiel*.
    :math:`y=\sum_{i=1}^{p} w_ix_i+b`. 
    :math:`z` est appelée la sortie du neurone.
    :math:`f` est appelée la fonction de transfert ou de seuil.
    :math:`z=f \pa{y} = f \pa {   \sum_{i=1}^{p} w_ix_i+b }`.

La réprésentation :ref:`graphique <fig-nn-neurone>` est plus souvent
celle qu'on retient. Ce schéma est également plus proche de sa définition 
biologique et dissocie mieux les rôles non symétriques 
des entrées et des poids. Des exemples de fonctions de transfert 
sont donnés par la table qui suit.
Les plus couramment utilisées sont les fonctions linéaire et sigmoïde.

.. cssclass:: table-hover

============================================= ======================================
exemples de fonction de transfert ou de seuil expression
============================================= ======================================
escalier                                      :math:`1_{\left[  0,+\infty\right[  }`
linéaire                                      :math:`x`
sigmoïde entre :math:`\cro{0,1}`              :math:`\dfrac{1}{1+e^{-x}}`
sigmoïde entre :math:`\cro{-1,1}`             :math:`1-\dfrac{2}{1+e^{x}}`
normale                                       :math:`e^{-\frac{x^{2}}{2}}`
exponentielle                                 :math:`e^{x}`
============================================= ======================================



La plupart des fonctions utilisées sont dérivables et cette propriété 
s'étend à tout assemblage de neurones, ce qui permet d'utiliser 
l'algorithme de rétropropagation découvert par 
[Rumelhart1986]_.
Ce dernier permet le calcul de la dérivée ouvre ainsi les portes 
des méthodes d'optimisation basées sur cette propriété.


Une couche de neurones
++++++++++++++++++++++

.. mathdef::
    :title: couche de neurones
    :tag: Définition
    :lid: rn_definition_couche_neurone_1
    
    Soit :math:`p` et :math:`n` deux entiers naturels, 
    on note :math:`W \in \R^{n\pa{p+1}} = \pa{W_1,\dots,W_n}`
    avec :math:`\forall i \in \intervalle{1}{n}, \; W_i \in \R^{p+1}`.
    Une couche de :math:`n` neurones et :math:`p` entrées est une fonction :
    
    .. math::
    
        F : \R^{n\pa{p+1}} \times \R^p \longrightarrow \R^n
        
    vérfifiant : 
    
    * :math:`\forall i \in \intervalle {1}{n}, \; f_i` est un neurone.
    * :math:`\forall W \in \R^{n\pa{p+1}} \times \R^p, \; F\pa{W,x} = \pa {f_1\pa{W_1,x}, \dots, f_n\pa{W_n,x}}`


Une couche de neurones représente la juxtaposition de plusieurs neurones 
partageant les mêmes entrées mais ayant chacun leur propre vecteur de 
coefficients et leur propre sortie.


Un réseau de neurones : le perceptron
+++++++++++++++++++++++++++++++++++++

.. mathdef::
    :title: réseau de neurones multi-couches ou perceptron
    :tag: Définition
    :lid: rn_definition_perpception_1

    Un réseau de neurones multi-couches à :math:`n` sorties, 
    :math:`p` entrées et :math:`C` couches est une liste de couches
    :math:`\vecteur{C_1}{C_C}` connectées les unes aux autres de telle sorte que :
    
    
    * :math:`\forall i \in \intervalle {1}{C}`, 
      chaque couche :math:`C_i` possède :math:`n_i` neurones et :math:`p_i` entrées
    * :math:`\forall i \in \intervalle{1}{C-1}, \; n_i = p_{i+1}`, 
      de plus :math:`p_1 = p` et :math:`n_C = n`
    
    Les coefficients de la couche :math:`C_i` sont notés 
    :math:`\pa {W_1^i,\dots,W_{n_i}^i}`, cette couche définit une fonction
    :math:`F_i`.
    Soit la suite :math:`\pa{Z_i}_{0\infegal i \infegal C}` définie par :
    
    .. math::
    
        \begin{array}{l}
        Z_0 \in \R^p \\
        \forall i \in \intervalle{1}{C}, \; Z_i = F_i \pa {W_1^i,\dots,W_{n_i}^i,Z_{i-1}}\end{array}

    On pose :math:`M = M = \sum_{i=1}^{C}n_i\pa{p_i+1}`, 
    le réseau de neurones ainsi défini est une fonction :math:`F` telle que :

    .. math::
    
        \begin{array}{lrll}
        F : & \R ^ M \times \R^p & \longrightarrow & \R^n \\
            & \pa{W,Z_0} & \longrightarrow & Z_C
        \end{array}


.. mathdef:: 
    :title: Modèle du perceptron multi-couche (multi-layer perceptron, MLP)
    :tag: Figure
    :lid: figure_peceptron-fig

    .. image:: rnimg/rn_gradient.png
        :height: 200
        
    * :math:`\vecteur{x_1}{x_p}` : entrées
    * :math:`C_i` nombre de neurones sur la couche :math:`i`, :math:`C_0 = p`
    * :math:`z_{c,i}` sortie du neurone :math:`i`, de la couche :math:`c`, par extension, :math:`z_{0,i} = x_i`
    * :math:`y_{c,i}` potentiel du neurone :math:`i` de la couche :math:`c`
    * :math:`w_{c,i,j}` coefficient associé à l'entrée :math:`j` du neurone :math:`i` de la couche :math:`c`
    * :math:`b_{c,i}` biais du neurone :math:`i` de la couche :math:`c`
    * :math:`f_{c,i}` fonction de seuil du neurone :math:`i` de la couche :math:`c`        
        
Souvent, on considère que les entrées forment la couche :math:`C_0` de 
manière à simplifier les écritures. Ainsi, 
chaque couche :math:`C_i` du perceptron a pour entrées les sorties 
de la couche :math:`C_{i-1}`. Cette définition est plus facile 
à illustrer qu'à énoncer (voir :ref:`Modèle du perceptron <figure_peceptron-fig>`)
et rappelle le rôle non symétrique des entrées et des poids. 
Le mécanisme qui permet de calculer les sorties d'un réseau de neurones 
sachant ses poids est appelé *propagation*.


.. mathdef:: 
    :title: Propagation
    :tag: Algorithme
    :lid: algo_propagation

    Cet algorithme s'applique à un réseau de neurones vérifiant la 
    définition du :ref:`perceptron`. Il s'agit
    de calculer les sorties de ce réseau connaissant ses poids 
    :math:`\pa{w_{c,i,j}}` et ses entrées :math:`\pa{x_j}`.
    
    *Initialisation*
    
    | for i in :math:`1..C_0` :
    |   z_{0,i} \longleftarrow x_i
    
    Vient ensuite le calcul itératif de la suite 
    :math:`\pa{Z_c}_{1 \infegal c \infegal C}` :
    
    *Récurrence*
    
    | for c in :math:`1..C` :
    |   for i in :math:`1..C_c` :
    |       :math:`z_{c,i} \longleftarrow 0`
    |       for j in :math:`1..C_{i-1}` :
    |           :math:`z_{c,i} \longleftarrow z_{c,i} + w_{c,i,j} z_{c-1,j}`
    |       :math:`z_{c,i} \longleftarrow f\pa { z_{c,i} + b_{c,i}}`

    
Le nombre de couches d'un réseau de neurones n'est pas limité. 
Les réseaux de deux couches (une couche pour les entrées, une couche de sortie) 
sont rarement utilisés. Trois couches sont nécessaires (une couche pour 
les entrées, une couche dite *cachée*, une couche de sortie) pour construire des 
modèles avec une propriété intéressante de densité :ref:`theoreme_densite`.


La régression
=============



.. [Rumelhart1986] Learning internal representations by error propagation (1986),
   D. E. Rumelhart, G. E. Hinton, R. J. Williams 
   in *Parallel distributed processing: explorations in the microstructures of cohniyionn MIT Press, Cambridge*

