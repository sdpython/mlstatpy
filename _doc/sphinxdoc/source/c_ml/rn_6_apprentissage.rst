

Apprentissage d'un réseau de neurones
=====================================

Le terme apprentissage est encore inspiré de la biologie et se traduit 
par la minimisation de la fonction :eq:`equation_fonction_erreur_g` où 
:math:`f` est un réseau de neurone défini par un :ref:`perceptron <rn_definition_perpception_1>`. 
Il existe plusieurs méthodes pour effectuer celle-ci. 
Chacune d'elles vise à minimiser la fonction d'erreur :

.. math::

        E\pa{W}   = G \pa{W}  =   \sum_{i=1}^{N} e\pa {Y_{i} - \widehat{Y_{i}^W}}
                                            =   \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W,X_{i}}}

Dans tous les cas, les différents apprentissages utilisent la suite 
suivante :math:`\pa{ \epsilon_{t}}` vérifiant :eq:`rn_suite_epsilon_train` 
et proposent une convergence vers un :ref:`minimum local <figure_modele_optimal>`.

.. math::
    :label: rn_suite_epsilon_train

    \forall t>0,\quad\varepsilon_{t}\in \R_{+}^{\ast} \text{ et }
    \sum_{t\geqslant0}\varepsilon_{t}=+\infty,\quad
    \sum_{t\geqslant0}\varepsilon_{t}^{2}<+\infty

Il est souhaitable d'apprendre plusieurs fois la même fonction en modifiant 
les conditions initiales de ces méthodes de manière à améliorer la robustesse de la solution.







Apprentissage avec gradient global
++++++++++++++++++++++++++++++++++


L'algorithme de :ref:`rétropropagation <algo_retropropagation>` permet d'obtenir 
la dérivée de l'erreur :math:`e` pour un vecteur d'entrée :math:`X`. Or l'erreur 
:math:`E\pa{W}` à minimiser est la somme des erreurs pour chaque exemple 
:math:`X_i`, le gradient global :math:`\partialfrac{E\pa{W}}{W}` de cette erreur 
globale est la somme des gradients pour chaque exemple 
(voir équation :eq:`algo_retro_1`). 
Parmi les méthodes d'optimisation basées sur le gradient global, on distingue deux catégories :

* Les méthodes du premier ordre, elles sont calquées sur la 
  `méthode de Newton <https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Newton>`_ 
  et n'utilisent que le gradient.
* Les méthodes du second ordre ou méthodes utilisant un 
  `gradient conjugué <https://fr.wikipedia.org/wiki/M%C3%A9thode_du_gradient_conjugu%C3%A9>`_
  elles sont plus coûteuses en calcul mais plus performantes
  puisque elles utilisent la dérivée seconde ou une valeur approchée.



.. rn_optim_premier_ordre:

Méthodes du premier ordre
^^^^^^^^^^^^^^^^^^^^^^^^^


\indexfrr{apprentissage}{premier ordre}
\indexfrr{méthode}{premier ordre} 
\indexfrr{ordre}{méthode du premier ordre}

Les méthodes du premier ordre sont rarement utilisées. 
Elles sont toutes basées sur le principe 
de la descente de gradient de Newton présentée dans 
la section :ref:`optimisation_newton` :

.. mathdef::
    :title: optimisation du premier ordre
    :lid: rn_algorithme_apprentissage_1
    :tag: Algorithme
    
    *Initialiation*
		
    Le premier jeu de coefficients :math:`W_0` du réseau 
    de neurones est choisi aléatoirement.
    
    .. math::
        
        \begin{array}{rcl}
        t   &\longleftarrow&    0 \\
        E_0 &\longleftarrow&    \sum_{i=1}^{N} e\pa {Y_{i} - f \pa{W_0,X_{i}}}
        \end{array}
    
    *Calcul du gradient*
    
    :math:`g_t \longleftarrow \partialfrac{E_t}{W} \pa {W_t} = \sum_{i=1}^{N} e'\pa {Y_{i} - f \pa{W_t,X_{i}}}
    
    *Mise à jour*
    
    .. math::
        
        \begin{array}{rcl}
        W_{t+1} &\longleftarrow& W_t - \epsilon_t g_t \\
        E_{t+1} &\longleftarrow& \sum_{i=1}^{N} e\pa {Y_i - f \pa{W_{t+1},X_i}} \\
        t       &\longleftarrow& t+1
        \end{array}

    
    *Terminaison*
    
    Si :math:`\frac{E_t}{E_{t-1}} \approx 1` (ou :math:`\norm{g_t} \approx 0`) 
    alors l'apprentissage a convergé sinon retour au calcul du gradient.


La condition d'arrêt peut-être plus ou moins stricte selon les besoins du problème. 
Cet algorithme converge vers un minimum local de la fonction d'erreur 
(d'après le théorème de :ref:`convergence <theoreme_convergence>` 
mais la vitesse de convergence est inconnue.


