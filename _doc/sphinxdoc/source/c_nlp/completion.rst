

.. _l-completion0:

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

Quelques éléments de codes sont disponibles dans le module
:mod:`completion <mlstatpy.nlp.completion>` et le notebook 
:ref:`completiontrierst`. Vous pouvez également lire 
`How to Write a Spelling Corrector <http://norvig.com/spell-correct.html>`_
de `Peter Norvig <http://norvig.com/>`_ et découvrir le sujet 
avec `On User Interactions with Query Auto-Completion <https://www.semanticscholar.org/paper/On-user-interactions-with-query-auto-completion-Mitra-Shokouhi/71e953caa2542a61b52e684649b3569c00251021/pdf>`_
de Bhaskar Mitra, Milad Shokouhi, Filip Radlinski, Katja Hofmann.


.. totree::
    :depth: 1
    
    completion_formalisation
    completion_fausse
    completion_metrique
    completion_propriete
    completion_optimisation
    completion_implementation
    completion_digression


Notebooks associés :

* :ref:`completiontrierst`
* :ref:`completionprofilingrst`
* :ref:`completiontrielongrst`

