Le perceptron multicouches est un type de réseau neuronal artificiel organisé en plusieur couches.
Un perceptron multicouche possède au moins trois couches : une couche d'entrée, au moins une couche cachée et une couche de sortie.
Chaque couche est constituée d'un nombre (potentiellement différent) de neuronnes.
L'information circule de la couche d'entrée vers la couche de sortie uniquement : il s'ajit donc d'un réseau à propagation directe (feedforward).
Les neurones de la dernière couche sont les sorties du système global.

Les premiers perceptrons étaient constitués d'une couche unique.
On pouvait calculer le OU logique avec un perceptron , par contre, ils étaient incapables de résoudre des problèmes non linéaires come le OU exclusif.
Cette limitation fut supprimée au travers de la rétropropagation du gradient de l'erreur dans les systèmes multicouches, proposé par Paul Werbos en 1974 et mis au point en 1986 par David Rumelhart.

Dans le perceptron multicouche à rétropropagation, les neurones d'une couche sont reliés à la totalité des neurones des couches adjacentes. Ces liaisons sont soumises à un coefficient altérant l'effet de l'information sur le neurone de destination. Ainsi le poids de chacune de ces liaison est l'élément clé du fonctionnement du réseau : la mise en place d'un perceptron multicouche pour résoudre un problème passe donc par la détermination des meilleurs poids applicables à chacune des connexions inter-neuronales.
