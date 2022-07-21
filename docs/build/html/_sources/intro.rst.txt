Introduction
============

 **MIRAG** : **M**\odule d' **I**\nversion et de **R**\econstruction par **A**\DMM pour le **G**\PR


Le stage réalisé au sein du LISTIC et en collaboration avec l'entreprise Géolithe dans le cadre du projet SMGA [1]_, a pour but la mise en œuvre de techniques de traitement des signaux afin d'exploiter le concept de GPR (**G**\round **P**\enetrating **R**\adar) aéroporté, cette démarche s'inscrit dans une optique de preuve de concept ou faisabilité.
 
Cette proposition est relativement nouvelle et profite de l'essor conjoint des technologies de drone et de l'intelligence artificielle. Ce stage s'articule autour de deux axes principaux : l'amélioration de la qualité des radargrammes obtenus et la classification de signaux d'intérêt au sein de ces derniers. Dans ce présent rapport nous nous concentrons principalement sur la présentation du premier axe, et présentons les pistes et méthodes du second axe qui seront utilisés pour la deuxième partie du stage.

Jusqu'à très récemment l'utilisation du GPR était souvent associée à des configurations où le radar réalisant la mesure était posé, avec l'antenne plaqué au sol. De cette manière le radar peut maximiser la propagation de l'onde dans le sol avec peu de pertes au niveau de l'interface entre le radar et le sol. Mais depuis l'essor des drones, ces derniers sont des pistes intéressantes pour ausculter des lieux à distances et difficile d'accès.
C'est dans ce contexte que l'entreprise Géolithe désire étudier la faisabilité de ces GPR aéroportés. afin d'en améliorer l'acquisition d'image radar pour l'étude de cavités.

L'intérêt est double puisqu'il s'agit d'automatiser le traitement et l'interprétation de radargrammes dans le contexte des radar aéroportés où l'identification des zones d'intérêt est rendue plus difficile par 2 composantes :

-   La stabilité et la position de l'appareil qui doivent être d'une grande précision pour permettre la construction de l'image sans artefact.

-   La couche d'air présente entre le radar et la surface du sol à étudiée, qui atténue fortement la propagation de l'onde et qui peut varier en fonction de la hauteur de l'appareil.

--------
Contexte
--------

Le principe du GPR est connu et principalement utilisé dans des domaines variés, des fouilles archéologiques en passant par la détection de mine ou la reconnaissance de marqueurs paléoclimatiques (Daniels [2]_,Tinelli [3]_). Cette méthode peut être schématisée de la manière suivante:

.. image:: screenshots/contexte.png
   :alt: alternate text
   :class: with-shadow

Le GPR plaqué au sol se déplace lentement dans une direction et envoie une onde électromagnétique. Cette dernière est réfléchie par tous les obstacles qu'elle rencontre jusqu'à son atténuation. Le GPR va ainsi enregistrer toutes les réflexions en fonction du temps à une position donnée pour former une trace appelée A-scan. L'enregistrement et la concaténation d'un grand nombre d'A-scan le long du déplacement du GPR donne naissance au radargramme appelé B-scan. 

Les interfaces ou changements de matériaux, apparaissent ainsi comme des traits, tandis que les cibles sont reconnaissables à la forme de l'hyperbole, paramétrée par les caractéristiques du milieu (permittivité, fréquence de l'onde) et due à la variation de la distance entre le GPR et la cible.
Durant ce stage, nous faisons l'hypothèse d'un radar de type mono-statique où l'antenne réceptrice et émettrice sont confondues dans le GPR.

----------------------
Organisation du module
----------------------

Afin de répondre aux problématique énoncées ci dessus, nous nous sommes basés sur les travaux de Terrasse [4]_ et Wohlberg [5]_. La solution de reconstruction et d'amélioration des radargrammes étant la modélisation de ce dernier par le produit de convolution entre un dictionnaire de motifs simple et des cartes de coefficients.

.. image:: screenshots/conv.png
   :alt: alternate text
   :class: with-shadow


Le module se divise en deux grandes catégories :

- dictionnaire : fonctions qui permettent la création du dictionnaire pour la résolution du problème d'inversion, et ce par 2 approches mathématique et physique.
- optmisation : fonctions qui réalisent l'inversion pour sortir les meilleures cartes de coefficients nécessaire pour la reconstruction.

A cela s'ajoute des fonctions complémentaires d'affichages, de calcul de métrique et de filtrage nécessaire au bon fonctionnement du module.

.. [1] Stratigraphie de Montagne par Géoradar Aéroporté.
.. [2] David Daniels. A review of gpr for landmine detection. Sensing and Imaging,7 :90–123, 09 2006.
.. [3] Chiara Tinelli, Adriano Ribolini, Giovanni Bianucci, Monica Bini, and Walter Landini. Ground penetrating radar and palaeontology : The detection of sirenian fossil bones under a sunflower field in tuscany (italy). Comptes Rendus Palevol,11(6) :445–454, 2012.
.. [4] Guillaume Terrasse. Géodétection des réseaux enterrés par imagerie radar. Theses,Télécom ParisTech, March 2017
.. [5] Brendt Wohlberg. Admm penalty parameter selection by residual balancing, 2017
