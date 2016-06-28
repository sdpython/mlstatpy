



Méthode d'optimisation de Newton
================================

Lorsqu'un problème d'optimisation n'est pas soluble de manière déterministe, 
il existe des algorithmes permettant de trouver une solution approchée 
à condition toutefois que la fonction à maximiser ou minimiser soit dérivable, 
ce qui est le cas des réseaux de neurones. Plusieurs variantes seront proposées 
regroupées sous le terme de descente de gradient.



Algorithme et convergence
+++++++++++++++++++++++++


Soit :math:`g : \R \dans \R` une fonction dérivable dont il faut trouver 
:math:`\overset{*}{x} = \underset{x \in \R}{\arg \min} \; g\pa{x}`, 
le schéma suivant illustre la méthode de descente de gradient 
dans le cas où :math:`g \pa{x} = x^2`.


.. image:: rnimg/rn_courbe.png


On note :math:`x_{t}` l'abscisse à l'itération :math:`t`.
On note :math:`\dfrac{\partial g\left(  x_{t}\right)  }{\partial x}` le
gradient de :math:`g\left(  x\right)  =x^{2}`.
L'abscisse à l'itération :math:`t+1` sera 
:math:`x_{t+1}=x_{t}-\varepsilon_{t}\left[  \dfrac{\partial g\left(  x_{t}\right)}{\partial x}\right]`.
:math:`\varepsilon_{t}` est le pas de gradient à l'itération :math:`t`.

On suppose maintenant que :math:`g` est une fonction dérivable 
:math:`g : \R^q \dans \R` dont il faut trouver le minimum, le théorème suivant démontre 
la convergence de l'algorithme de descente de gradient à condition 
que certaines hypothèses soient vérifiées. Une généralisation de ce théorème est présentée dans
[Driancourt1996]_.


.. mathdef::
    :title: convergence de la méthode de Newton [Bottou1991]_
    :tag: Théorème
    :lid: theoreme_convergence


    Soit une fonction continue :math:`g : W \in \R^M \dans \R`
    de classe :math:`C^{1}`.    
    On suppose les hypothèses suivantes vérifiées :
    
    * **H1** : :math:`\underset{W\in \R^q}{\arg\min} \; 
      g\left(  W\right) =\left\{  W^{\ast}\right\}` 
      est un singleton
    * **H2** : :math:`\forall\varepsilon>0, \; \underset{\left|  W-W^{\ast}\right|
      >\varepsilon}{\inf}\left[  \left(  W-W^{\ast}\right)  ^{\prime}.\nabla
      g\left(  W\right)  \right]  >0`
    * **H3** : :math:`\exists\left(  A,B\right)  \in \R^2` tels que :math:`\forall W\in\R^p,\; \left\|
      \nabla g\left( W\right) \right\| ^{2}\leqslant A^{2}+B^{2}\left\|  W-W^{\ast}\right\|  ^{2}`
    * **H4** : la suite :math:`\left(  \varepsilon_{t}\right)_{t\geqslant0}` vérifie,
      :math:`\forall t>0, \; \varepsilon_{t}\in \R_{+}^{\ast}`
      et :math:`\sum_{t\geqslant 0}\varepsilon_{t}=+\infty`, 
      :math:`\sum_{t\geqslant 0}\varepsilon_{t}^{2}<+\infty`
    
    Alors la suite :math:`\left(  W_{t}\right)  _{t\geqslant 0}` construite de la manière suivante 
    :math:`W_{0} \in \R^M`, :math:`\forall t\geqslant0` : 
    :math:`W_{t+1}=W_{t}-\varepsilon_{t}\,\nabla g\left(  W_{t}\right)`            
    vérifie :math:`\lim_{ t \dans+\infty}W_{t}=W^{\ast}`.



L'hypothèse **H1** implique que le minimum de la fonction :math:`g` 
est unique et l'hypothèse **H2** implique que le demi-espace défini par 
l'opposé du gradient contienne toujours le minimum de la fonction :math:`g`. 
L'hypothèse **H3** est vérifiée pour une fonction sigmoïde, elle l'est donc aussi pour toute somme finie
de fonctions sigmoïdes que sont les réseaux de neurones à une couche cachée.



**Démonstration du théorème**

*Partie 1*


Soit la suite :math:`u_{t}=\ln\left(  1+\varepsilon_{t}^{2}x^{2}\right)` 
avec :math:`x\in\R`, comme :math:`\sum_{t\geqslant 0} \varepsilon_{t}^{2} < +\infty, \; 
u_{t}\thicksim\varepsilon_{t}^{2}x^{2}`, on a :math:`\sum_{t\geqslant 0} u_{t} < +\infty`.

Par conséquent, si :math:`v_{t}=e^{u_{t}}` alors :math:`\prod_{t=1}^T v_{t}\overset{T \rightarrow \infty}{\longrightarrow}D \in \R`.

*Partie 2*

On pose :math:`h_{t}=\left\|  W_{t}-W^{\ast}\right\|  ^{2}`.
Donc :

.. math::
    :nowrap:
    :label: equation_convergence_un

    \begin{eqnarray}
    h_{t+1} -h_{t} &=&\left\|  W_{t}-\varepsilon_{t}\,\nabla g\left( W_{t}\right) -W^{\ast }\right\|
    			  ^{2}-\left\|W_{t}-W^{\ast}\right\| ^{2}
    \end{eqnarray}

Par conséquent :

.. math::

    h_{t+1}-h_{t}=-2\varepsilon_{t}\underset{>0} {\underbrace{\left(  W_{t}-W^{\ast}\right) 
     ^{\prime}\,\nabla g\left( W_{t}\right)
    }}+\varepsilon_{t}^{2}\,\left\|  \,\nabla C\left( W_{t}\right) \right\|  
    ^{2}\leqslant\varepsilon_{t}^{2}\,\left\|  \,\nabla g\left( W_{t}\right)
    \right\|  ^{2}\leqslant\varepsilon_{t}^{2}\,\left(  A^{2}  +B^{2}h_{t}\right)
    
D'où :

.. math::

    h_{t+1}-h_{t}\left(  1+\varepsilon_{t}^{2}B^{2}\right) \leqslant\varepsilon_{t}^{2}\,A^{2}
    
On pose :math:`\pi_{t}= \prod_{k=1}^t \left(  1+\varepsilon_{k}^{2}B^{2}\right)  ^{-1}` 
alors en multipliant des deux côtés par :math:`\pi_{t+1}`, on obtient :

.. math::

    \begin{array}{rcl}
    \pi_{t+1}h_{t+1}-\pi_{t}h_{t} &\leqslant& \varepsilon_{t}^{2}\,A^{2}\pi_{t+1}\\
    \text{d'où }\pi_{q+1}h_{q+1}-\pi_{p}h_{p} &\leqslant&
                    \sum_{t=p}^q \varepsilon_{t}^{2}\,A^{2}\pi_{t+1} \leqslant
    \sum_{t=p}^{q} \varepsilon_{t}^{2} \, A^{2}\Pi  \leqslant \sum_{t=p}^{q} \varepsilon_{t}^{2}\,A^{2}\Pi
    			 \underset{t \longrightarrow
    \infty}{\longrightarrow} 0
    \end{array}

Comme la série :math:`\sum_t \pa{\pi_{t+1}h_{t+1}-\pi_{t}h_{t}}` vérifie le critère de Cauchy, elle est convergente. Par conséquent :
    
.. math::

    \underset{q\rightarrow\infty}{\lim}\pi_{q+1}h_{q+1}=0=\underset{q\rightarrow \infty}{\lim}\Pi h_{q+1}
    
D'où :math:`\underset{q\rightarrow\infty}{\lim}h_{q}=0`.

*Partie 3*


La série :math:`\sum_t\pa{h_{t+1}-h_{t}}` est convergente car :math:`\Pi h_t \sim \pi_t h_t`.
:math:`\sum_{t\geqslant0}\varepsilon_{t}^{2}\,\left\| \,\nabla g\left( W_{t}\right) \right\|  ^{2}` 
l'est aussi (d'après **H3**).

D'après :eq:`equation_convergence_un`, 
la série :math:`\sum_{t\geqslant 0}\varepsilon_{t}\left( W_{t}-W^{\ast }\right) ^{\prime} \,
\nabla g\left( W_{t}\right)` est donc convergente. 
Or d'après les hypothèses **H2**, **H4**, elle ne peut l'être que si :
    
.. math::
    :nowrap:

    \begin{eqnarray}
    \underset{t\rightarrow\infty}{\lim}W_{t}&=&W^{\ast}
    \end{eqnarray}



Si ce théorème prouve la convergence 
de la méthode de Newton, il ne précise pas à quelle vitesse cette convergence 
s'effectue et celle-ci peut parfois être très lente. Plusieurs variantes 
ont été développées regroupées sous le terme de méthodes de quasi-Newton dans le but 
d'améliorer la vitesse de convergence (voir :ref:`rn_section_train_rn`).

Ce théorème peut être étendu dans le cas où la fonction :math:`g` 
n'a plus un seul minimum global mais plusieurs minima locaux ([Bottou1991]_), 
dans ce cas, la suite :math:`\pa{W_{t}}` converge vers un mimimum local. 
Dans le cas des réseaux de neurones, la fonction à optimiser est :

.. math::
    :nowrap:
    :label: equation_fonction_erreur_g
    
    \begin{eqnarray}
    G\pa{W}   &=&   \sum_{i=1}^{N} e\pa {Y_{i}, \widehat{Y_{i}^W}} \\
                      &=&   \sum_{i=1}^{N} e\pa {Y_{i}, f \pa{W,X_{i}}}
    \end{eqnarray}

Dès que les fonctions de transfert ne sont pas linéaires,
il existe une multitude de minima locaux, ce nombre croissant avec celui des coefficients.




