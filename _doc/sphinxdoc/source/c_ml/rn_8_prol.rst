

Prolongements
=============



Base d'apprentissage et base de test
++++++++++++++++++++++++++++++++++++



Les deux exemples de régression et de classification
:ref:`rn_section_regression` et :ref:`subsection_classifieur` ont montré 
que la structure du réseau de neurones la mieux adaptée a 
une grande importance. Dans ces deux cas, une rapide vérification visuelle 
permet de juger de la qualité du modèle obtenu après apprentissage, 
mais bien souvent, cette "vision" est inaccessible pour 
des dimensions supérieures à deux. Le meilleur moyen de jauger 
le modèle appris est de vérifier si l'erreur obtenue sur une base 
ayant servi à l'apprentissage (ou *base d'apprentissage*) est conservée 
sur une autre base (ou *base de test*) que le modèle découvre pour la première fois.

Soit :math:`B=\acc{\pa{X_i,Y_i} | 1 \infegal i \infegal N}` 
l'ensemble des observations disponibles. Cet ensemble est 
aléatoirement scindé en deux sous-ensembles :math:`B_a` et :math:`B_t`  
de telle sorte que :

.. math::

    \begin{array}{l}
    B_a \neq \emptyset \text{ et } B_t \neq \emptyset \\
    B_a \cup B_t = B \text{ et } B_a \cap B_t = \emptyset \\
    \frac{\#{B_a}}{\#{B_a \cup B_t}} = p \in ]0,1[ 
    			\text{, en règle générale, } p \in \cro{\frac{1}{2},\frac{3}{4}}
    \end{array}

Ce découpage est valide si tous les exemples de la base :math:`B` 
obéissent à la même loi, les deux bases :math:`B_a` et :math:`B_t` 
sont dites *homogènes*. Le réseau de neurones sera donc appris sur la 
base d'apprentissage :math:`B_a` et "testé" sur la base de test 
:math:`B_t`. Le test consiste à vérifier que l'erreur sur :math:`B_t` 
est sensiblement égale à celle sur :math:`B_a`, auquel cas on dit que le 
modèle (ou réseau de neurones) généralise bien. Le modèle trouvé 
n'est pas pour autant le bon modèle mais il est robuste. 
La courbe figure suivante illustre une définition du modèle optimal 
comme étant celui qui minimise l'erreur sur la base de test. 
Lorsque le modèle choisi n'est pas celui-là, deux cas sont possibles :


* Le nombre de coefficients est trop petit : 
  le modèle généralise bien mais il existe d'autres modèles 
  meilleurs pour lesquels l'erreur d'apprentissage et de test est moindre.
* Le nombre de coefficients est trop grand : le modèle généralise mal, 
  l'erreur d'apprentissage est faible et l'erreur de test élevée, 
  le réseau a appris la base d'apprentissage par coeur.


.. mathdef::
    :title: Modèle optimal pour la base de test
    :tag: Figure
    
    .. image:: rnimg/errapptest.png
    
Ce découpage des données en deux bases d'apprentissage et de 
test est fréquemment utilisé pour toute estimation de modèles 
résultant d'une optimisation réalisée au moyen d'un algorithme itératif. 
C'est le cas par exemple des modèles de Markov cachés. 
Elle permet de s'assurer qu'un modèle s'adapte bien à de nouvelles données.


.. _rnn_fonction_base_radiale_rbf:

Fonction de transfert à base radiale
++++++++++++++++++++++++++++++++++++

La fonction de transfert est dans ce cas à base radiale 
(souvent abrégée par RBF pour `radial basis function <https://en.wikipedia.org/wiki/Radial_basis_function>`_.
Elle ne s'applique pas au produit scalaire entre le 
vecteur des poids et celui des entrées mais 
à la distance euclidienne entre ces vecteurs.

.. mathdef::
    :title: neurone distance
    :lid: rn_definition_neurone_dist
    :tag: Définition

    Un neurone distance à :math:`p` entrées est une fonction 
    :math:`f : \R^{p+1} \times \R^p \longrightarrow \R` définie par :
    
    * :math:`g : \R \dans \R`
    * :math:`W \in \R^{p+1}`, :math:`W=\pa{w_1,\dots,w_{p+1}} = \pa{W',w_{p+1}}`
    * :math:`\forall x \in \R^p, \; f\pa{W,x} = e^{-\norm{W'-x}^2 + w_{p+1}}`
      avec :math:`x = \pa{x_1,\dots,x_p}`


Ce neurone est un cas particulier du suivant qui pondère chaque 
dimension par un coefficient. Toutefois, ce neurone possède :math:`2p+1` 
coefficients où :math:`p` est le nombre d'entrée.


.. mathdef::
    :title: neurone distance pondérée
    :tag: Définition
    :lid: rn_definition_neurone_dist_pond
    
    Pour un vecteur donné :math:`W \in \R^p = \pa{w_1,\dots,w_p}`, 
    on note :math:`W_i^j = \pa{w_i,\dots,w_j}`.
    Un neurone distance pondérée à :math:`p` entrées est une fonction 
    :math:`f : \R^{2p+1} \times \R^p \longrightarrow \R` définie par :
    
    * :math:`g : \R \dans \R`
    * :math:`W \in \R^{2p+1}`, :math:`W=\pa{w_1,\dots,w_{2p+1}} = \pa{w_1,w_{2p+1}}`
    * :math:`\forall x \in \R^p, \; f\pa{W,x} = 
      \exp \cro {-\cro{\sum_{i=1}^{p} w_{p+i}\pa{w_i - x_i}^2 } + w_{p+1}}`
      avec :math:`x = \pa{x_1,\dots,x_p}`


La fonction de transfert est :math:`x \longrightarrow e^x` 
est le potentiel de ce neurone donc : 
:math:`y = -\cro{\sum_{i=1}^{p} w_{p+i}\pa{w_i - x_i}^2 } + w_{p+1}`.

L'algorithme de :ref:`rétropropagation <algo_retropropagation>`
est modifié par l'insertion d'un tel neurone dans un réseau ainsi que la rétropropagation. 
Le plus simple tout d'abord :

.. math::
    :nowrap:
    :label: eq_no_distance_nn

    \begin{eqnarray*}
    1 \infegal i \infegal p, & \dfrac{\partial y}{\partial w_{i}} = & - 2 w_{p+i}\pa{w_i - x_i} \\  
    p+1 \infegal i \infegal 2p, & \dfrac{\partial y}{\partial w_{i}} = & - \pa{w_i - x_i}^2 \\  
    i = 2p+1, & \dfrac{\partial y}{\partial w_{i}} = & -1
    \end{eqnarray*}
    
Pour le neurone distance simple, la ligne :eq:`eq_no_distance_nn` 
est superflue, tous les coefficients :math:`(w_i)_{p+1 \infegal i \infegal 2p}` 
sont égaux à 1. La relation :eq:`retro_eq_nn_3` reste vraie mais n'aboutit plus à:eq:`algo_retro_5`, 
celle-ci devient en supposant que la couche d'indice :math:`c+1` 
ne contient que des neurones définie par la définition précédente.


.. math::
    :nowrap:
    
    \begin{eqnarray*}
    \partialfrac{e}{y_{c,i}}  
                                &=& \sum_{l=1}^{C_{c+1}}              \partialfrac{e}{y_{c+1,l}}
                                                                    \partialfrac{y_{c+1,l}}{z_{c,i}}
                                                                    \partialfrac{z_{c,i}}{y_{c,i}}  \\
         &=& \cro{ \sum_{l=1}^{C_{c+1}}              
         						\partialfrac{e}{y_{c+1,l}}
                    \pa{ 2 w_{c+1,l,p+i} \pa{ w_{c+1,l,i} - z_{c,i} } } }
                    \partialfrac{z_{c,i}}{y_{c,i}} 
    \end{eqnarray*}





Poids partagés
++++++++++++++



Les poids partagés sont simplement un ensemble de poids qui sont 
contraints à conserver la même valeur. Soit :math:`G` un groupe de poids 
partagés dont la valeur est :math:`w_{G}`. Soit :math:`X_k` et :math:`Y_k` 
un exemple de la base d'apprentissage (entrées et sorties désirées), 
l'erreur commise par le réseau de neurones est :math:`e\left(  W,X_k,Y_k\right)`.

.. math::

    \dfrac{\partial e\left(  W,X_{k},Y_{k}\right)  }
    {\partial w_{G}}=\sum_{w\in G}\dfrac{\partial e\left(  W,X_{k},Y_{k}\right) }{\partial
    w_G}\dfrac{\partial w_{G}}{\partial w}=\sum_{w\in G}
    {\sum} \dfrac{\partial e\left(  W,X_{k},Y_{k}\right)  }{\partial w_G}

Par conséquent, si un poids :math:`w` appartient à un groupe :math:`G` de poids partagés, 
sa valeur à l'itération suivante sera :

.. math::

    w_{t+1}=w_{t}-\varepsilon_{t}\left(  \underset{w\in G}
    {\sum}\dfrac{\partial e\left(  W,X_{k},Y_{k}\right)  }{\partial w}\right)


Cette idée est utilisée dans les 
`réseaux neuronaux convolutifs <https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif>`_
(`deep learning <https://fr.wikipedia.org/wiki/Apprentissage_profond>`_,
`CS231n Convolutional Neural Networks for Visual Recognition <http://cs231n.github.io/neural-networks-1/#layers>`_).


Dérivée par rapport aux entrées
+++++++++++++++++++++++++++++++


On note :math:`\left(  X_k,Y_k\right)` un exemple de la base d'apprentissage. 
Le réseau de neurones est composé de :math:`C` couches, :math:`C_i` est le 
nombre de neurones sur la ième couche, :math:`C_0` est le nombre d'entrées. 
Les entrées sont appelées :math:`\left( z_{0,i}\right) _{1\leqslant i\leqslant C_{0}}$, $\left(  y_{1,i}\right)  _{1\leqslant i\leqslant C_{1}}` 
sont les potentiels des neurones de la première couche, on en déduit que, dans le cas d'un neurone classique (non distance) :

.. math:: 

		\dfrac{\partial e\left(  W,X_{k},Y_{k}\right)  }{\partial z_{0,i}} =
			\underset{j=1}{\overset{C_{1}}{\sum}}\dfrac{\partial e\left(  W,X_{k}
		,Y_{k}\right)  }{\partial y_{1,j}}\dfrac{\partial y_{1,j}}{\partial z_{0,i}
		 }=\underset{j=1}{\overset{C_{1}}{\sum}}\dfrac{\partial e\left( W,X_{k}
		,Y_{k}\right)  }{\partial y_{1,j}}w_{1,j,i}

Comme le potentiel d'un neurone distance n'est pas linéaire par 
rapport aux entrées :math:`\left( y=\overset{N} {\underset{i=1}{\sum}}\left( w_{i}-z_{0,i}\right)  ^{2}+b\right)`, 
la formule devient dans ce cas :

.. math:: 

		\dfrac{\partial e\left(  W,X_{k},Y_{k}\right)  }{\partial z_{0,i}} =
				\underset{j=1}{\overset{C_{1}}{\sum}}\dfrac{\partial e\left(  W,X_{k}
		,Y_{k}\right)  }{\partial y_{1,j}}\dfrac{\partial y_{1,j}}{\partial z_{0,i}
			 }=-2\underset{j=1}{\overset{C_{1}}{\sum}}\dfrac{\partial e\left(
		W,X_{k},Y_{k}\right)  }{\partial y_{1,j}}\left(  w_{1,j,i}-z_{0,i}\right)






.. _rn_decay:

Régularisation ou Decay
+++++++++++++++++++++++


Lors de l'apprentissage, comme les fonctions de seuil du réseau de 
neurones sont bornées, pour une grande variation des coefficients, 
la sortie varie peu. De plus, pour ces grandes valeurs, la dérivée 
est quasi nulle et l'apprentissage s'en trouve ralenti. Par conséquent, 
il est préférable d'éviter ce cas et c'est pourquoi un terme de 
régularisation est ajouté lors de la mise à jour des 
coefficients (voir [Bishop1995]_). L'idée consiste à ajouter 
à l'erreur une pénalité fonction des coefficients du réseau de neurones :
:math:`E_{reg} = E + \lambda \; \sum_{i} \; w_i^2`.

Et lors de la mise à jour du poids :math:`w_i^t` à l'itération :math:`t+1` :
:math:`w_i^{t+1} = w_i^t - \epsilon_t \cro{ \partialfrac{E}{w_i} - 2\lambda w_i^t }`.

Le coefficient :math:`\lambda` peut décroître avec le nombre 
d'itérations et est en général de l'ordre de :math:`0,01` pour un 
apprentissage avec gradient global, plus faible pour un 
apprentissage avec gradient stochastique.


Problèmes de gradients
++++++++++++++++++++++

La descente du gradient repose sur l'algorithme de :ref:`rétropropagation <algo_retropropagation>`
qui propoge l'erreur depuis la dernière couche jusqu'à la première.
Pour peu qu'une fonction de seuil soit saturée. Hors la zone rouge, 
le gradient est très atténué. 

.. plot::

    import matplotlib.pyplot as plt
    import numpy
    def softmax(x):
        return 1.0 / (1 + numpy.exp(-x))
    def dsoftmax(x):
        t = numpy.exp(-x)
        return t / (1 + t)**2
    x = numpy.arange(-10,10, 0.1)
    y = softmax(x)
    dy = dsoftmax(x)
    fig, ax = plt.subplots(1,1)
    ax.plot(x,y, label="softmax")
    ax.plot(x,dy, label="dérivée")
    ax.set_ylim([-0.1, 1.1])
    ax.plot([-5, -5], [-0.1, 1.1], "r")
    ax.plot([5, 5], [-0.1, 1.1], "r")
    ax.legend(loc=2)
    plt.show()

.. index:: vanishing gradient problem 

Après deux couches de fonctions de transferts, le
gradient est souvent diminué. On appelle ce phénomène
le `Vanishing gradient problem <https://en.wikipedia.org/wiki/Vanishing_gradient_problem>`_.
C'est d'autant plus probable que le réseau est gros. Quelques pistes pour y remédier :
`Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients <http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/>`_,
`Why are deep neural networks hard to train? <http://neuralnetworksanddeeplearning.com/chap5.html>`_.
L'article `Deep Residual Learning for Image Recognition <http://arxiv.org/pdf/1512.03385v1.pdf>`_
présente une structure de réseau qui va dnas le même sens.
De la même manière, la norme du gradient peut exploser plus particulièrement dans le cas des 
`réseaux de neurones récurrents <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ : 
`Understanding the exploding gradient problem <http://arxiv.org/pdf/1211.5063v1.pdf>`_.


.. selection_connexion:


Sélection de connexions
+++++++++++++++++++++++


Ce paragraphe présente un algorithme de sélection de l'architecture 
d'un réseau de neurones proposé par Cottrel et Al. dans [Cottrel1995]_. 
La méthode est applicable à tout réseau de neurones mais n'a été démontrée 
que pour la classe de réseau de neurones utilisée pour la 
:ref:`régression <nn-regression>`. Les propriétés qui suivent ne sont 
vraies que des réseaux à une couche cachée et dont les sorties 
sont linéaires. Soit :math:`\pa{X_k,Y_k}` un exemple de la base 
d'apprentissage, les résidus de la régression sont supposés normaux 
et i.i.d. L'erreur est donc (voir :ref:`rn_enonce_probleme_regression`) :
:math:`e\left( W,X_k,Y_k\right) =\left(f\left( W,X_k\right)  -Y_k\right)^2`.

On peut estimer la loi asymptotique des coefficients du réseau de neurones. 
Des connexions ayant un rôle peu important peuvent alors être supprimées 
sans nuire à l'apprentissage en testant la nullité du coefficient associé. 
On note :math:`\widehat{W}` les poids trouvés par apprentissage et 
:math:`\overset{\ast}{W}` les poids optimaux. On définit :

.. math::
    :nowrap:
    :label: rn_selection_suite

    \begin{eqnarray*}
    \text{la suite } \widehat{\varepsilon_{k}} &=&   f\left(  \widehat{W} ,X_{k}\right)  -Y_{k}, \;
    							 \widehat{\sigma}_{N}^{2}=\dfrac{1}{N}\underset
                                    {k=1}{\overset{N}{\sum}}\widehat{\varepsilon_{k}}^{2} \\
    \text{la matrice }
    \widehat{\Sigma_{N}}      &=&   \dfrac{1}{N}\left[  \nabla_{\widehat{W}%
                                    }e\left(  W,X_{k},Y_{k}\right)  \right]  
                                    \left[  \nabla_{\widehat{W}}
                                    e\left(  W,X_{k},Y_{k}\right)  \right]  ^{\prime}
    \end{eqnarray*}


.. mathdef::
    :title: loi asymptotique des coefficients
    :lid: theoreme_loi_asym
    :tag: Théorème

    Soit :math:`f` un réseau de neurone défini par :ref:`perceptron <rn_definition_perpception_1>` 
    composé de :

    * une couche d'entrées
    * une couche cachée dont les fonctions de transfert sont sigmoïdes
    * une couche de sortie dont les fonctions de transfert sont linéaires
    
    Ce réseau sert de modèle pour la fonction :math:`f` 
    dans le problème de :ref:`régression <problem-regression>` 
    avec un échantillon :math:`\vecteur{\pa{X_1,Y_1}}{\pa{X_N,Y_N}}`, 
    les résidus sont supposés normaux.
    La suite :math:`\pa{\widehat{\epsilon_k}}` définie par :eq:`rn_selection_suite` vérifie :
    
    .. math::
    
        \dfrac{1}{N} \sum_{i=1}^{N} \widehat{\epsilon_k} = 0 = \esp\cro{f\pa{\widehat{W},X} - Y}
    
    Et le vecteur aléatoire :math:`\widehat{W} - W^*` vérifie :
    
    .. math::
    
        \sqrt{N} \cro { \widehat{W} - W^* } \; \overset{T \rightarrow + \infty}{\longrightarrow} \;
                \loinormale{0}{\widehat{\sigma_N}^2  \widehat{\Sigma_N}}
    
    Où la matrice :math:`\widehat{\Sigma_N}` est définie par :eq:`rn_selection_suite`.
    
    \end{xtheorem}

.. mathdef::
    :title: Réseau de neurones pour lequel la sélection de connexions s'applique
    :lid: figure_selection_connexion_reseau-fig
    :tag: Figure
    
    .. image:: rnimg/selection_connexion.png


La démonstration de ce théorème est donnée par l'article [Cottrel1995]_. 
Ce théorème mène au corollaire suivant :

.. mathdef::
    :title: nullité d'un coefficient
    :tag: Corollaire
		
    Les notations utilisées sont celles du théorème sur :ref:`loi asymptotique des coefficients <theoreme_loi_asym>`. 
    Soit :math:`w_k` un poids du réseau de neurones
    d'indice quelconque :math:`k`. Sa valeur estimée est :math:`\widehat{w_k}`, 
    sa valeur optimale :math:`w^*_k`. D'après le théorème :
    
    .. math::
    
        N \dfrac{ \pa{\widehat{w_k} - w^*_k}^2  } { \widehat{\sigma_N}^2 \pa{\widehat{\Sigma_N}^{-1}}_{kk} }
        \; \overset{T \rightarrow + \infty}{\longrightarrow} \; \chi^2_1

Ce résultat permet, à partir d'un réseau de neurones, de supprimer les 
connexions pour lesquelles l'hypothèse de nullité n'est pas réfutée. 
Afin d'aboutir à l'architecture minimale adaptée au problème, 
Cottrel et Al. proposent dans [Cottrel1995]_ l'algorithme suivant :

.. mathdef::
    :title: sélection d'architecture
    :lid: rn_algorithme_selection_connexion_1
    :tag: Théorème

    Les notations utilisées sont celles du théorème 
    :ref:`loi asymptotique des coefficients <theoreme_loi_asym>`. 
    :math:`f` est un réseau de neurones
    de paramètres :math:`W`. On définit la constante :math:`\tau`, 
    en général :math:`\tau = 3,84` puisque 
    :math:`\pr {X < \tau} = 0,95` si :math:`X \sim \chi_1^2`.
    
    *Initialisation*
    
    Une architecture est choisie pour le réseau de neurones :math:`f` incluant un nombre `M` de paramètres.
    
    *Apprentissage*
    
    Le réseau de neurones :math:`f` est appris. On calcule les nombre et matrice 
    :math:`\widehat{\sigma_N}^2` et :math:`\widehat{\Sigma_N}`. 
    La base d'apprentissage contient :math:`N` exemples.
    
    *Test*
    
    | for :math:`k` in :math:`1..M`
    |   :math:`t_k \longleftarrow N \dfrac{ \widehat{w_k} ^2  } { \widehat{\sigma_N}^2 \pa{\widehat{\Sigma_N}^{-1}}_{kk} }`
    
    *Sélection*
    
    | :math:`k' \longleftarrow \underset{k}{\arg \min} \; t_k`
    | si :math:`t_{k'} < \tau`
    |   Le modèle obtenu est supposé être le modèle optimal. L'algorithme s'arrête.
    | sinon
    |   La connexion :math:`k'` est supprimée ou le poids :math:`w_{k'}` est maintenue à zéro.
    |   :math:`M \longleftarrow M-1`
    |   Retour à l'apprentissage.


Cet algorithme est sensible au minimum local trouvé lors de l'apprentissage, il est préférable d'utiliser des méthodes
du second ordre afin d'assurer une meilleure convergence du réseau de neurones.

L'étape de sélection ne supprime qu'une seule connexion. Comme l'apprentissage
est coûteux en calcul, il peut être intéressant de supprimer toutes les connexions 
:math:`k` qui vérifient :math:`t_k < \tau`. Il est toutefois conseillé de ne 
pas enlever trop de connexions simultanément puisque la suppression d'une connexion nulle peut
réhausser le test d'une autre connexion, nulle à cette même itération, mais non nulle à l'itération suivante.
Dans l'article [Cottrel1995]_, les auteurs valident leur algorithme dans le cas d'une 
régression grâce à l'algorithme suivant.

.. mathdef::
    :title: validation de l'algorithme de sélection des coefficients
    :lid: nn_algorithme_valid_selection
    :tag: Algorithme
    
    *Choix aléatoire d'un modèle*
    
    Un réseau de neurones est choisi aléatoirement,  
    soit :math:`f : \R^p \dans \R` la fonction qu'il représente.
	Une base d'apprentissage :math:`A` (ou échantillon) 
    de :math:`N` observations est générée aléatoirement à partir de ce modèle :
      
    .. math::
        
        \begin{array}{l}
        \text{soit } \pa{\epsilon_i}_{1 \infegal i \infegal N} \text{ un bruit blanc} \\
        A = \acc{ \left. \pa{X_i,Y_i}_{1 \infegal i \infegal N} \right| 
                    \forall i \in \intervalle{1}{N}, \; Y_i = f\pa{X_i} + \epsilon_i }
        \end{array}
        
    *Choix aléatoire d'un modèle*
		
    L'algorithme de :ref:`sélection <rn_algorithme_selection_connexion_1>` 
    à un réseau de neurones plus riche que le modèle choisi
    dans l'étape d'initilisation. Le modèle sélectionné est noté :math:`g`.
		
	*Validation*
    
	Si :math:`\norm{f-g} \approx 0`,
    l'algorithme de :ref:`sélection <rn_algorithme_selection_connexion_1>` est validé.
		
		
