


Démonstration du théorème de la densité des réseaux de neurones
===============================================================

.. index:: rn_enonce_probleme_regression

Formulation du problème de la régression
++++++++++++++++++++++++++++++++++++++++


Soient deux variables aléatoires continues 
:math:`\pa{X,Y} \in \R^p \times \R^q \sim \loi` quelconque, 
la résolution du problème de :ref:`régression <problem-regression>` 
est l'estimation de la fonction :math:`\esp(Y|X) = F\pa{X}`.
Pour cela, on dispose d'un ensemble de points 
:math:`A = \acc{ \pa{X_{i},Y_{i}} \sim \loi | 1 \infegal i \infegal N }`.
    
Soit :math:`f : \R^M \times \R^p \longrightarrow \R^q` une fonction, on définit 
:math:`\forall i \in \intervalle{1}{N}, \; \widehat{Y_{i}^{W}} = f \pa{W,X_{i}}`.
:math:`\widehat{Y_{i}^{W}}` est appelée la valeur prédite pour `X_{i}`.
On pose alors 
:math:`\epsilon_{i}^{W} = Y_{i} -  \widehat{Y_{i}^{W}} = Y_{i} - f \pa{W,X_{i}}`.

Les résidus sont supposés 
`i.i.d. (identiquement et indépendemment distribués) <https://fr.wikipedia.org/wiki/Variables_ind%C3%A9pendantes_et_identiquement_distribu%C3%A9es>`_,
et suivant une loi normale 
:math:`\forall i \in \intervalle{1}{N}, \; \epsilon_{i}^{W} \sim \loinormale{\mu_{W}}{\sigma_{W}}`
La vraisemblance d'un échantillon
:math:`\pa{Z_i}_{1\infegal i \infegal N}`, 
où les :math:`Z_i` sont indépendantes entre elles et suivent la loi de densité 
:math:`f \pa{z | \theta}` 
est la densité du vecteur :math:`\vecteur{Z_1}{Z_N}` qu'on exprime 
comme suit :

.. math::

    \begin{array}{rrcl}
                    &L\pa{\theta, \vecteurno{Z_1}{Z_N}} & =& \prod_{n=1}^{N} f\pa{Z_i | \theta} \\
    \Longrightarrow&
    \ln L\pa{\theta, \vecteurno{Z_1}{Z_N}} &=& \sum_{n=1}^{N} \ln f\pa{Z_i | \theta}
    \end{array}
    

La log-vraisemblance de l'échantillon s'écrit
:math:`L_{W} = -\frac{1}{2\sigma_{W}^2} \sum_{i=1}^{N}
\pa{Y_{i} - \widehat{Y_{i}^W} - \mu_{W} }^2 + N\ln\pa{\sigma_{W}\sqrt{2\pi}}`.
Les estimateurs du maximum de vraisemblance 
pour :math:`\mu_W` et :math:`\sigma_W` sont (voir [Saporta1990]_) :


.. math::

    \begin{array}{rcl}
    \widehat{\mu_{W}}     &=&     \frac{1}{N} \sum_{i=1}^{N} Y_{i} - \widehat{Y_{i}^W} \\
    \widehat{\sigma_{W}}  &=&     \sqrt{ \frac{ \sum_{i=1}^{N} \pa{Y_{i} - 
                                  \widehat{Y_{i}^W} - \mu_{W}}^2}{N}}
    \end{array}


L'estimateur de :math:`\widehat{Y}=f\pa{W,X}` désirée est de préférence 
sans biais (:math:`\mu_W = 0`) et de variance minimum, 
par conséquent, les paramètres :math:`\overset{*}{W}` 
qui maximisent la vraisemblance :math:`L_W` sont :


.. math::
    :label: rn_eqn_regression_1

    \begin{array}{rcl}
    \overset{*}{W}   &=& \underset{W \in \R^M}{\arg \min} \sum_{i=1}^{N} 
                                            \pa {Y_{i} - \widehat{Y_{i}^W}}^2 \\
                     &=& \underset{W \in \R^M}{\arg \min} \sum_{i=1}^{N} 
                            \pa {Y_{i} - f \pa{W,X_{i}}}^2
    \end{array}


Réciproquement, on vérifie que si :math:`W^*` vérifie 
l'équation :eq:`rn_eqn_regression_1` alors l'estimateur défini par 
:math:`f` est sans biais
Il suffit pour s'en convaincre de poser 
:math:`g = f + \alpha` avec 
:math:`\alpha \in \R` et de vérifier que la valeur optimale pour 
:math:`\alpha` est 
:math:`\alpha = - \frac{1}{N}\, \sum_{i=1}^{N} \, \left. Y_i - f\pa{W,X_i} \right.`.
L'estimateur minimise la vraisemblance :math:`L_W`. 
Cette formule peut être généralisée en faisant une autre hypothèse 
que celle de la normalité des résidus (l'indépendance étant conservée), 
l'équation :eq:`rn_eqn_regression_1`
peut généralisée par :eq:`rn_eqn_regression_2`.

.. math::
    :label: rn_eqn_regression_2
    
    \begin{array}{rcl}
    \overset{*}{W}     &=& \underset{W \in \R^M}{\arg \min} \sum_{i=1}^{N} 
                                                            e\pa {Y_{i} - \widehat{Y_{i}^W}} \\
                        &=& \underset{W \in \R^M}{\arg \min} \sum_{i=1}^{N} 
                                e\pa{Y_{i} - f \pa{W,X_{i}}} 
    \end{array}

Où la fonction :math:`e : \R^q \in \R` est appelée fonction d'erreur.







Densité des réseaux de neurones
+++++++++++++++++++++++++++++++



L'utilisation de réseaux de neurones s'est considérablement 
développée depuis que l'algorithme de rétropropagation a 
été trouvé ([LeCun1985]_, [Rumelhart1986]_, [Bishop1995]_). 
Ce dernier permet d'estimer la dérivée d'un réseau de neurones en 
un point donné et a ouvert la voie à des méthodes classiques 
de résolution pour des problèmes d'optimisation tels que la régression non linéaire.

Comme l'ensemble des fonctions polynômiales, 
l'ensemble des fonctions engendrées par des réseaux de neurones 
multi-couches possède des propriétés de :ref:`densité <theoreme_densite>`
et sont infiniment dérivables. Les réseaux de neurones comme 
les polynômes sont utilisés pour modéliser la fonction 
:math:`f` de l'équation :eq:`rn_eqn_regression_2`.
Ils diffèrent néanmoins sur certains points

Si une couche ne contient que des fonctions de transfert bornées 
comme la fonction sigmoïde, tout réseau de neurones incluant cette couche 
sera aussi borné. D'un point de vue informatique, il est 
préférable d'effectuer des calculs avec des valeurs du même 
ordre de grandeur. Pour un polynôme, les valeurs des termes de 
degré élevé peuvent être largement supérieurs à leur somme.

Un autre attrait est la symétrie dans l'architecture d'un réseau 
de neurones, les neurones qui le composent jouent des rôles 
symétriques (corollaire :ref:`familles libres <corollaire_famille_libre>`. 
Pour améliorer l'approximation d'une fonction, dans un cas, 
il suffit d'ajouter un neurone au réseau, dans l'autre, 
il faut inclure des polynômes de degré plus élevé que ceux déjà  employés.


.. mathdef::
    :title: densité des réseaux de neurones (Cybenko1989)
    :lid: theoreme_densite
    :tag: Théorème

    [Cybenko1989]_
    Soit :math:`E_{p}^{q}` l'espace des réseaux de neurones à 
    :math:`p` entrées et :math:`q` sorties, possédant une couche cachée dont la
    fonction de seuil est une fonction sigmoïde 
    :math:`\left(  x\rightarrow 1-\frac{2}{1+e^{x}}\right)`,
    une couche de sortie dont la fonction de seuil est linéaire 
    Soit :math:`F_{p}^{q}` l'ensemble des fonctions continues de 
    :math:`C\subset\R^{p}\longrightarrow\R^{q}` avec :math:`C` 
    compact muni de la norme 
    :math:`\left\| f\right\| =\underset{x\in C}{\sup}\left\|  f\left( x\right)  \right\|`
    Alors :math:`E_{p}^{q}` est dense dans :math:`F_{p}^{q}`.
			

La démonstration de ce théorème nécessite deux lemmes. 
Ceux-ci utilisent la définition usuelle du produit scalaire
sur :math:`\R^p` défini par
:math:`\pa{x,y} = \pa{\vecteurno{x_1}{x_p},\vecteurno{y_1}{y_p}} \in \R^{2p} \longrightarrow
\left\langle x,y \right\rangle = \sum_{i=1}^{p} x_i y_i`.
et la norme infinie : 
:math:`x = \vecteur{x_1}{x_p} \in \R^p \longrightarrow \norm{x} = 
\underset{i \in \intervalle{1}{p}}{\max} x_i`.
Toutes les normes sont 
`équivalentes <https://fr.wikipedia.org/wiki/Norme_%C3%A9quivalente>`_ 
sur :math:`\R^p`.




.. mathdef::
    :title: approximation d'une fonction créneau
    :lid: theoreme_densite_lemme_a
    :tag: Corollaire

    Soit :math:`C \subset \R^p, \; C= \acc { \vecteur{y_1}{y_p} \in \R^p \, | \forall i\in \intervalle{1}{p},\, 0 \leqslant y_{i}\leqslant 1   }`, 
    alors :
    
    .. math::
    
        \begin{array}{l}
        \forall \varepsilon > 0, \; \forall \alpha>0, \; \exists n \in \N^*, \; 
                    \exists \vecteur{x_1}{x_n} 
                    \in\left(  \R^p\right)  ^{n}, \; \exists 
            \vecteur{\gamma_1}{\gamma_n} \in \R^n  \text{ tels que } \forall x\in \R^p, \\ \\
        \begin{array}{ll}
        &   \left| \underset{i=1}{\overset{n}{\sum}}\dfrac{\gamma_i}
                        {1+e^{\left\langle x_{i},x\right\rangle +b_{i}}}-\indicatrice{x\in C
            }\right| \leqslant1 \\ \\
        \text{ et } &   \underset{y\in Fr\left( C\right)  }{\inf }\left\| x-y\right\| > 
                        \alpha\Rightarrow\left| \underset{i=1}{\overset
            {n}{\sum}}\dfrac{\gamma_i}{1+e^{\left\langle x_{i},x\right\rangle +b_{i}}} 
                    -\indicatrice{x\in C}\right| \leqslant\varepsilon
        \end{array}
        \end{array}
		
		
**Démonstration du corollaire**

*Partie 1*

Soit :math:`h` la fonction définie par : 
:math:`h\pa{x} = \pa{\dfrac{1}{1+e^{-kx}}}^p` 
avec :math:`p>0` et :math:`0 < \epsilon < 1`.
A :math:`\alpha`, :math:`\epsilon` fixé, :math:`0 < \epsilon < 1`, 
on cherche :math:`k` tel que :
    
.. math::

    \begin{array}{crcl}
                    &   \epsilon                    &=& h\pa{\alpha} = \pa{\dfrac{1}{1+e^{-k\alpha}}}^p \\
    \Longrightarrow &   \epsilon^{-\frac{1}{p}}               &=& 1+e^{-k\alpha} \\
    \Longrightarrow &   \epsilon^{-\frac{1}{p}} -1            &=& e^{-k\alpha} \\
    \Longrightarrow &   \ln \pa{\epsilon^{-\frac{1}{p}} -1}   &=& -k\alpha \\
    \Longrightarrow &   k                           &=& - \dfrac{ \ln\pa{\epsilon^{-\frac{1}{p}} -1}}{\alpha} =
                                                            k_0\pa{\epsilon,\alpha,p}
    \end{array}

*Partie 2*


Soit :math:`\alpha>0` et :math:`1\geqslant\varepsilon>0, \, k>0`,

On pose :math:`f\left(  y_{1},...,y_{p}\right)  =\underset{i=1}{\overset{p}{\prod}}
\dfrac{1}{1+e^{-ky_{i}}}\underset{i=1}{\overset{p}{\prod}}\dfrac {1}{1+e^{-k\left(  1-y_{i}\right)}}`
d'après sa définition, :math:`0 \infegal f\left(  y_{1},...,y_{p}\right)  \infegal 1`.

Pour :math:`k \supegal k_0 \pa{\epsilon,\alpha,2p}` 
obtenu dans la partie précédente :

.. math::

    \underset{_{i\in\left\{ 1,...,p\right\}}}{\inf} 
    \cro { \min\left\{  \left|  y_{i}\right|  ,\left|  1-y_{i}\right|  \right\} } >\alpha  
    \Longrightarrow\left\|  f\left(  y_{1},...,y_{p}\right) - \indicatrice{x\in C}\right\|  \infegal\varepsilon

*Partie 3*

Soit :math:`g` la fonction définie par :

.. math::

    \begin{array}{rcl}
    g\pa{x}     &=&     \pa{\dfrac{1}{1+e^{-kx}}}\pa{\dfrac{1}{1+e^{-k\pa{1-x}}}} 
                =     \dfrac{1}{1+e^{-kx}+e^{-k\pa{1-x}}+e^{-k}} \\ 
                &=&     \dfrac{1}{1+e^{-kx}+e^{-k}e^{kx}+e^{-k}} 
                =     \dfrac{e^{kx}}{e^{kx}\pa{1+e^{-k}}+1+e^{-k}e^{2kx}}
    \end{array}

La fonction :math:`x \longrightarrow e^{kx}\pa{1+e^{-k}}+1+e^{-k}e^{2kx}` 
est un polynôme en :math:`e^{kx}` dont le
discriminant est positif. Par conséquent la fraction 
rationnelle :math:`g\pa{x}` admet une décomposition en éléments
simples du premier ordre 
et il existe quatre réels :math:`\eta_1`, :math:`\eta_2`, 
:math:`\delta_1`, :math:`\delta_2` tels que :

.. math::

    g\pa{x} = \dfrac{\eta_1}{1+ e^{kx+\delta_1}} + \dfrac{\eta_2}{1+ e^{kx+\delta_2}}

Par conséquent :

.. math::

    f\vecteur{y_1}{y_p} = \prod_{i=1}^{p} g\pa{y_i} =
                          \prod_{i=1}^{p} \cro { \dfrac{\eta_1^i}{1+ e^{ky_i+\delta_1^i}} + \dfrac{\eta_2^i}{1+
                          e^{ky_i+\delta_2^i}} }

Il existe :math:`n \in \N` tel qu'il soit possible d'écrire :math:`f` sous la forme :

.. math::

    f\pa{y} = \sum_{i=1}^{n}  \dfrac{\gamma_i}{ 1 + e^{ <x_i,y> + b_i } }



.. mathdef::
    :title: approximation d'une fonction indicatrice
    :lid: theoreme_densite_lemme_b
    :tag: Corollaire

    Soit :math:`C\subset\R^p` compact, alors : 
    
    .. math::

        \begin{array}{c}
        \forall\varepsilon>0, \; \forall\alpha>0, \; \exists\left(  x_{1},...,x_{n}\right) 
                \in\left(  \R^{p}\right)^{n}, \; \exists\left(
        b_{1},...,b_{n}\right)  \in\R^n \text{ tels que } \forall x\in\R^{p},\\ \\
        \begin{array}{ll}
        &   \left|  \sum_{i=1}^n \dfrac{\gamma_i}
                    {1+e^{\left\langle x_{i},x\right\rangle +b_{i}}}-\indicatrice{x\in C
            }\right|  \leqslant1+2\varepsilon^2\\ \\
        \text{ et } &   \underset{y\in Fr\left( C\right)  }{\inf}\left\|  x-y\right\|
            >\alpha\Rightarrow\left| \sum_{i=1}^n 
                        \dfrac{\gamma_i}{1+e^{\left\langle x_{i} ,x\right\rangle +b_{i}}}-
            \indicatrice{x\in C}\right| \leqslant \varepsilon
        \end{array}
        \end{array}

**Démonstration du corollaire**

*Partie 1*


Soit :math:`C_1=\left\{  y=\left(  y_{1},...,y_{p}\right)  \in\R^p
\,\left| \, \forall i\in\left\{  1,...,n\right\}  ,\,0\leqslant y_{i}\leqslant1\right.  \right\}`
et :math:`C_{2}^{j}=\left\{  y=\left(
y_{1},...,y_{p}\right)  \in\R^p\,\left| \,
\forall i\neq j,\,0\leqslant y_{i}\leqslant1 \text{ et }1\leqslant y_{j}\leqslant2\right.
\right\}`

Le premier lemme suggère que la fonction cherchée pour ce lemme 
dans le cas particulier :math:`C_1\cup C_2^j` est :

.. math::

    \begin{array}{rcl}
    f\left(  y_{1},...,y_{p}\right) &=&   \prod_{i=1}^p \dfrac
                                        {1}{1+e^{-ky_{i}}} \prod_{i=1}^p\dfrac{1}{1+e^{-k\left( 1-y_{i}\right)
                                        }}+ \\
                                &&      \quad \left(  \prod_{i \neq j}
                                        \dfrac
                                        {1}{1+e^{-ky_{i}}}\right)  \left(  \prod_{i \neq j}
                                        \dfrac{1}{1+e^{-k\left(  1-y_{i}\right)  }}\right)
                                        \dfrac{1}{1+e^{k\left( 1-y_{j}\right)  }}\dfrac{1}{1+e^{-k\left(  2-y_{j}\right)
                                        }}\\
    %
                                &=&  \left(  \prod_{i \neq j} \dfrac{1}{1+e^{-ky_{i}}}\right)
                                    \left(  \prod_{i \neq j} \dfrac{1}{1+e^{-k\left(  1-y_{i}\right)
                                    }}\right) \\
                                &&  \quad  \left( \dfrac{1}{1+e^{-ky_{j}}}\dfrac{1}{1+e^{-k\left(  1-y_{j}\right)  }}
                                     +\dfrac {1}{1+e^{k\left(  1-y_{j}\right)  }}
                                                \dfrac{1}{1+e^{-k\left(2-y_{j}\right) }}\right)
                                     \\
    %
                                &=& \left(  \prod_{i \neq j} \dfrac{1}{1+e^{-ky_{i}}}\right)
                                     \left(  \prod_{i \neq j} \dfrac{1}{1+e^{-k\left(  1-y_{i}\right)  }}\right) \\
                                &&  \quad \left[\dfrac{1}{1+e^{-ky_{j}}}\left(  \dfrac{1}{1+e^{-k\left(  1-y_{j}\right)  }
                                    }+1-1\right)  +\left(  1-\dfrac{1}{1+e^{-k\left(  1-y_{j}\right)  }}\right)
                                    \dfrac{1}{1+e^{-k\left(  2-y_{j}\right)  }}\right]
    \end{array}


Pour :math:`k \supegal k_0\pa{\epsilon,\alpha,2p}`, on a :

.. math::

    \begin{array}{rcl}
    f\left(  y_{1},...,y_{p}\right)  &=& \left(  \prod_{i\neq j}
    \dfrac{1}{1+e^{-ky_{i}}}\right)  \left(  \prod_{i\neq j}
    \dfrac{1}{1+e^{-k\left(  1-y_{i}\right)  }}\right)
    \\
    && \quad \left(  \dfrac{1}%
    {1+e^{-ky_{j}}}+\dfrac{1}{1+e^{-k\left(  2-y_{j}\right)  }}+
    \underset {\leqslant\varepsilon^{2}}{\underbrace{\dfrac{1}{1+e^{k\left( 1-y_{j}\right)
    }}\dfrac{1}{1+e^{-ky_{j}}}}}-\underset{\leqslant\varepsilon^{2}}%
    {\underbrace{\dfrac{1}{1+e^{-k\left(  1-y_{j}\right)  }}\dfrac{1}%
    {1+e^{-k\left(  2-y_{j}\right)  }}}}\right)
    \end{array}

Par conséquent, il est facile de construire la fonction cherchée 
pour tout compact connexe par arc.

*Partie 2*

Si un compact :math:`C` n'est pas connexe par arc, 
on peut le recouvrir par une somme finie de
compacts connexes par arcs et disjoints 
:math:`\left(C_{k}\right) _{1\leqslant k\leqslant K}` de telle sorte que :

.. math::

    \forall y\in\underset{k=1}{\overset{K}{\cup}}C_{k},\,\inf\left\{  \left\|
    x-y\right\|  ,\,x\in C\right\}  \leqslant\dfrac{\alpha}{2}



**Démontration du théorème de** :ref:`densité des réseaux de neurones <theoreme_densite>`

*Partie 1*


On démontre le théorème dans le cas où :math:`q=1`.
Soit :math:`f` une fonction continue du compact 
:math:`C\subset\R^p\rightarrow \R` et soit :math:`\varepsilon>0`.

On suppose également que :math:`f` est positive, dans le cas contraire, on pose 
:math:`f=\underset{\text{fonction positive}}{\underbrace{f-\inf f}}+\inf f`.

Si :math:`f` est nulle, alors c'est fini, sinon, on pose :math:`M=\underset{x\in C}{\sup }f\left(  x\right)`. 
:math:`M` existe car :math:`f` est continue et :math:`C` 
est compact (de même, :math:`\inf f` existe également).

On pose :math:`C_{k}=f^{-1}\left(  \left[  k\varepsilon,M\right]  \right)`. 
:math:`C_k` est compact car il est l'image
réciproque d'un compact par une fonction continue et :math:`C_k\subset C` compact.

.. image:: rnimg/rn_densite_idee.png


Par construction, :math:`C_{k+1}\subset C_{k}` et :math:`C=\underset{k=0}{\overset {\frac{M}{\varepsilon}}
{\bigcup}}C_{k}=C_{0}` on définit~:

.. math::

    \forall x\in
    C,\; g_{\varepsilon}\left(  x\right)  =
            \varepsilon\overset{\frac {M}{\varepsilon}}{ \sum_{k=0}}\indicatrice{x\in C_{k}}
  
D'où~:

.. math::
    :nowrap:
  
    \begin{eqnarray}
    f\left(  x\right)  -g_{\varepsilon}\left(  x\right)  &=& 
                        f\left(  x\right)-\varepsilon\overset{\frac{M}{\varepsilon}}{\sum_{k=0}}
        \indicatrice{x\in C_{k}} \nonumber 
    = f\left(  x\right)  -\varepsilon \overset{\frac{M}{\varepsilon}}
                {\sum_{k=0}}\indicatrice
                    { f\pa{x} \supegal k \varepsilon } \nonumber \\
    &=& f\left( x\right)  -\varepsilon\left[  \dfrac{f\left(  x\right) }
                    {\varepsilon}\right] \quad \text{ (partie entière)}\nonumber  \\
    & \text{d'où }&  0\leqslant f\left(  x\right)  -g_{\varepsilon}\left(  x\right)  \leqslant \frac{\varepsilon}{4}
    \end{eqnarray}


Comme :math:`f` est continue sur un compact, elle est uniformément continue sur ce compact :

.. math::

    \begin{array}{l}
    \exists\alpha>0 \text{ tel que } \forall\left(  x,y\right)  \in C^{2},
                \; \left\| x-y\right\|  \leqslant\alpha\Longrightarrow\left|  f\left(
        x\right) -f\left(  y\right)  \right|  \leqslant \frac{ \varepsilon}{2} \\ \\
    \text{ d'où } \left|  f\left(  x\right)  -f\left(  y\right)  \right| \supegal \varepsilon
                     \Longrightarrow\left\|  x-y\right\|  >\alpha
    \end{array}

Par conséquent :

.. math::

    \inf\left\{  \left\|  x-y\right\|  \,\left|  \,x\in Fr\left(  C_{k}\right) ,\,y\in 
                    Fr\left(  C_{k+1}\right)  \right.  \right\}
    >\alpha

D'après le second lemme, on peut construire des fonctions :math:`h_{k}\left( x\right)
=\sum_{i=1}^n\dfrac{1}{1+e^{\left\langle x_{i}^{k},x\right\rangle +b_{i}^{k}}}` 
telles que :

.. math::

    \left(  \left\|  h_{k}\left(  x\right)  -\indicatrice{x\in C_{k}}\right\|  
        \leqslant1 \right)  \text{ et } \left( \underset{y\in
    Fr\left(  C\right)  }{\inf}\left\|  x-y\right\|  >\dfrac{\alpha}{2}%
    \Rightarrow\left\|  h_{k}\left(  x\right)  -\indicatrice{x\in C_{k}}\right\|  \leqslant\varepsilon^{2}\right)

On en déduit que :

.. math::

    \begin{array}{rcl}
    \left|  f\left(  x\right)  -\varepsilon\overset{\frac{M}{\varepsilon}}
            {\sum_{k=0}}h_{k}\left(  x\right)  \right|  &\leqslant&
        \left| f\left(  x\right)  -g_{\varepsilon}\left(  x\right)  \right| 
             +\left|g_{\varepsilon}\left(  x\right)  -\varepsilon
        \overset{\frac{M}{\varepsilon}}{\sum_{k=0}}h_{k}\left(  x\right)  \right| \\
    &\leqslant& \varepsilon+ \varepsilon^2 \left[  \dfrac{M}{\varepsilon}\right] + 2\varepsilon^2 \\
    &\leqslant& \varepsilon\left(  M+3\right)
    \end{array}

Comme :math:`\varepsilon\overset{\frac{M}{\varepsilon}}{\sum_{k=1}}
h_{k}\left(  x\right)` est de la forme désirée, le théorème est démontré dans le cas :math:`q=1`.


*Partie 2*


Dans le cas :math:`q>1`, on utilise la méthode précédente pour chacune des projections de :math:`f`
dans un repère orthonormé de :math:`\R^{q}`. Il suffit de
sommer sur chacune des dimensions.





Ce théorème montre qu'il est judicieux de modéliser la fonction 
:math:`f` dans l'équation :eq:`rn_eqn_regression_2` 
par un réseau de neurones puisqu'il possible de s'approcher d'aussi 
près qu'on veut de la fonction :math:`\esp\pa{Y | X}`, 
il suffit d'ajouter des neurones sur la couche cachée du réseau. 
Ce théorème permet de déduire le corollaire suivant :

.. mathdef::
    :title: famille libre de fonctions
    :tag: Corollaire
    :lid: corollaire_famille_libre

    Soit :math:`F_{p}` l'ensemble des fonctions continues de 
    :math:`C\subset\R^{p}\longrightarrow\R` avec :math:`C`
    compact muni de la norme :
    :math:`\left\| f\right\| =\underset{x\in C}{\sup}\left\|  f\left( x\right)  \right\|`
    Alors l'ensemble :math:`E_{p}` des fonctions sigmoïdes :
    
    .. math::
      
      E_{p} =  \acc{ x \longrightarrow 1 - \dfrac{2}{1 + e^{<y,x>+b}} | y 
      \in \R^p \text{ et } b \in \R}
    
    est une base de :math:`F_{p}`.


**Démonstration du corollaire**


Le théorème de :ref:`densité <theoreme_densite>` montre que la famille 
:math:`E_{p}` est une famille génératrice. Il reste à montrer que c'est une 
famille libre. Soient :math:`\pa{y_i}_{1 \infegal i \infegal N} \in \pa{\R^p}^N` et 
:math:`\pa{b_i}_{1 \infegal i \infegal N} \in \R^N` vérifiant :
:math:`i \neq j \Longrightarrow y_i \neq y_j \text{ ou } b_i \neq b_j`.
Soit :math:`\pa{\lambda_i}_{1 \infegal i \infegal N} \in \R^N`, il faut montrer que :

.. math::
    :nowrap:
    :label: corollaire_demo_recurrence_base
    
    \begin{eqnarray}
    \forall x \in \R^p, \; \sum_{i=1}^{N} \lambda_i \pa{ 1 - \dfrac{2}{1 + e^{<y_i,x>+b_i}  }} = 0
    \Longrightarrow \forall i \, \lambda_i = 0 
    \end{eqnarray}
 
C'est évidemment vrai pour :math:`N=1`. 
La démonstration est basée sur un raisonnement par récurrence, 
on suppose qu'elle est vraie pour :math:`N-1`, 
démontrons qu'elle est vraie pour :math:`N`. 
On suppose donc :math:`N \supegal 2`. 
S'il existe :math:`i \in \ensemble{1}{N}` tel que :math:`y_i = 0`, 
la fonction :math:`x \longrightarrow 1 - \dfrac{2}{1 + e^{<y_i,x>+b_i}}` 
est une constante, par conséquent, dans ce cas le corollaire est 
est vrai pour :math:`N`. Dans le cas contraire, 
:math:`\forall i \in \ensemble{1}{N}, \; y_i \neq 0`. 
On définit les vecteurs :math:`X_i = \pa{x_i,1}` et 
:math:`Y_i = \pa{y_j, b_j}`. 
On cherche à résoude le système de :math:`N` équations à :math:`N` inconnues :

.. math::
    :nowrap:
    :label: rn_coro_eq_1

    \begin{eqnarray}
    \left\{
    \begin{array}{ccc}
    \sum_{j=1}^{N} \lambda_j \pa{ 1 - \dfrac{2}{1 + e^{<Y_j,X_1>}}} &=& 0 \\
    \ldots \\
    \sum_{j=1}^{N} \lambda_j \pa{ 1 - \dfrac{2}{1 + e^{<Y_j,X_i>}}} &=& 0 \\
    \ldots \\
    \sum_{j=1}^{N} \lambda_j \pa{ 1 - \dfrac{2}{1 + e^{<Y_j,X_N>}}} &=& 0
    \end{array}
    \right.
    \end{eqnarray}
 
On note le vecteur 
:math:`\Lambda = \pa{\lambda_i}_{ 1 \infegal i \infegal N}` et :math:`M` la matrice :
 
.. math::

    M= \pa{m_{ij}}_{ 1 \infegal i,j \infegal N} = \pa{ 1 - \dfrac{2}{1 + e^{<Y_j,X_i>}} }_{ 1 \infegal i,j \infegal N}
 
L'équation :eq:`rn_coro_eq_1` est équivalente à l'équation matricielle : 
:math:`M\Lambda = 0`. On effectue une itération du pivot de Gauss.
:eq:`rn_coro_eq_1` équivaut à :
 
.. math::
    
    \begin{array}{rcl}
    &\Longleftrightarrow& \left\{ \begin{array}{ccllllllll}
                                    \lambda_1  m_{11} &+& \lambda_2 & m_{12} &+& \ldots &+& \lambda_N & m_{1N} & = 0 \\
                                    0                 &+& \lambda_2 & \pa{ m_{22} m_{11} - m_{12} m_{21} } 
                                    									&+& \ldots &+& \lambda_N & \pa{ m_{2N} m_{11} - m_{1N} m_{21} }
                                    									 & = 0 \\
                                    \ldots \\
                                    0                 &+& \lambda_2 & \pa{ m_{N2} m_{11} - m_{12} m_{N1} } &+& \ldots 
                                    									&+& \lambda_N & \pa{ m_{NN} m_{11} - m_{1N} m_{N1} } & = 0
                                    \end{array}
                                    \right. 
    \end{array}
 
On note :math:`\Lambda_* = \pa{\lambda_i}_{ 2 \infegal i \infegal N}` et 
:math:`\Delta_*`, :math:`M_*` les matrices :
 
.. math::

    \begin{array}{rcl}
    M_*         &=&     \pa{m_{ij}}_{ 2 \infegal i,j \infegal N} \\
    \Delta_*    &=&     \pa{ m_{1j} \, m_{i1} }_{ 2 \infegal i,j \infegal N}
    \end{array}
 
Donc :eq:`rn_coro_eq_1` est équivalent à :

.. math::
    :nowrap:
    :label: rn_coro_eq_3
 
    \begin{eqnarray}
    \begin{array}{ccl}
                         &\Longleftrightarrow& \left\{ \begin{array}{cccc}
                                    \lambda_1  m_{11}&+& \lambda_2  m_{12} + \ldots + \lambda_N  m_{1N}  &= 0 \\
                                    0                &+&   \pa{ m_{11} M_* -  \Delta_*} \Lambda_* & = 0
                                    \end{array}
                                    \right.
    \end{array}
    \end{eqnarray}
 
 
Il est possible de choisir :math:`X_1\pa{\alpha} = \pa{\alpha x_1, 1}` 
de telle sorte qu'il existe une suite :math:`\pa{s_l}_{ 1 \infegal l \infegal N } \in \acc{-1,1}^{N}`  
avec :math:`s_1=1` et vérifiant :

.. math::

    \forall j \in \vecteur{1}{N}, \; 
    \underset{\alpha \longrightarrow +\infty} {\lim }  \cro{ 1 - \dfrac{2}{1 + e^{<Y_j, \, X_1\pa{\alpha}   >}} } = 
    \underset{\alpha \longrightarrow +\infty} {\lim }  m_{1j}\pa{\alpha} = s_j
 
On définit :

.. math::

    \begin{array}{rll}
    U_* &=& \vecteur{m_{21}}{m_{N1}}' \\
    V_* &=& \vecteur{s_2 \, m_{21}}{s_N \, m_{N1}}' \\
    \text{ et la matrice } L_* &=& \pa{V_*}_ { 2 \infegal i \infegal N } \text{ dont les $N-1$ colonnes sont identiques }
    \end{array}
    
On vérifie que :

.. math::

		\underset{\alpha \longrightarrow +\infty} {\lim } \Delta\pa{\alpha} = V_*
 
On obtient, toujours pour :eq:`rn_coro_eq_1` :
 
 .. math::
    :nowrap:
    :label: rn_coro_eq_2
 
    \begin{eqnarray}
                         &\Longleftrightarrow& \left\{ \begin{array}{cclc}
                                    \lambda_1  m_{11}\pa{\alpha}	&+& 
                                    							\lambda_2  m_{12}\pa{\alpha} + \ldots + \lambda_N  m_{1N}\pa{\alpha}  &= 0 \\
                                    0                &+&   \cro{m_{11}\pa{\alpha} M_* -   
                                    													\pa{ L_* + \pa{ \Delta_*\pa{\alpha} - L_* } } } 
                                    												\Lambda_* & = 0
                                    \end{array}
                                    \right. \\ \nonumber\\
                         &\Longleftrightarrow& \left\{ \begin{array}{cclc}
                                    \lambda_1  m_{11}\pa{\alpha}	&+& 
                                    							\lambda_2  m_{12}\pa{\alpha} + \ldots + \lambda_N  m_{1N}\pa{\alpha}  &= 0 \\
                                    0                &+&   \pa{m_{11}\pa{\alpha} M_* -    L_* }      \Lambda_*
                                                         +  \pa{ \Delta_*\pa{\alpha} - L_* }     \Lambda_* &  = 0
                                    \end{array}
                                    \right. \nonumber
    \end{eqnarray}
 
On étudie la limite lorsque :math:`\alpha \longrightarrow +\infty` :
 
.. math::

    \begin{array}{crcl}
                        & \pa{ \Delta_*\pa{\alpha} - L_* }   &   
                        	\underset{ \alpha \rightarrow +\infty}{ \longrightarrow} & 0                 \\
    \Longrightarrow     & \pa{m_{11}\pa{\alpha} M_* -   L_* }      \Lambda_* &   
                            \underset{ \alpha \rightarrow +\infty}{ \longrightarrow} &  0\\
    \Longrightarrow     & \pa{M_* -  L_* }      \Lambda_* &   = &  0\\
    \Longrightarrow     & M_* \Lambda_* -    \pa{  \sum_{j=2}^{N} \lambda_j   }   V_*   &   = &  0\\
    \end{array}
    
Donc :

.. math::
    :nowrap:
    :label: rn_coro_eq_5
 
    \begin{eqnarray}
    M_* \Lambda_* -    \pa{  \sum_{j=2}^{N} \lambda_j   }   V_*   &   = &  0 \nonumber
    \end{eqnarray}
    
D'après l'hypothèse de récurrence, :eq:`rn_coro_eq_5` implique que : 
:math:`\forall i \in \ensemble{2}{N}, \; \lambda_i = 0`. 
Il reste à montrer que :math:`\lambda_1` 
est nécessairement nul ce qui est le cas losque :math:`\alpha \longrightarrow +\infty`, 
alors :math:`\lambda_1  m_{11}\pa{\alpha} \longrightarrow \lambda_1 = 0`. 
La récurrence est démontrée.
    
A chaque fonction sigmoïde du corollaire :ref:`famille libre <corollaire_famille_libre>` 
correspond un neurone de la couche cachée. Tous ont des rôles 
symétriques les uns par rapport aux autres ce qui ne serait 
pas le cas si les fonctions de transfert étaient des polynômes. 
C'est une des raisons pour lesquelles les réseaux de neurones 
ont du succès. Le théorème :ref:`densité <theoreme_densite>` 
et le corollaire :ref:`famille libre <corollaire_famille_libre>` 
sont aussi vraies pour des fonctions du type exponentielle : 
:math:`\pa{y,b} \in \R^p \times \R \longrightarrow e^{-\pa{<y,x>+b}^2}`. 
Maintenant qu'il est prouvé que les réseaux de neurones conviennent 
pour modéliser :math:`f` dans l'équation :eq:`rn_eqn_regression_2`, 
il reste à étudier les méthodes qui permettent de trouver 
les paramètres :math:`W^*` optimaux de cette fonction.





