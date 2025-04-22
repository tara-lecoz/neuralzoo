# Contexte du projet


# Veille

## Réseaux de Neurones Convolutifs (CNN) : Synthèse

### 1. Veille, Architecture Typique et Hyperparamètres

#### Veille (Contexte)

Les **Réseaux de Neurones Convolutifs** (Convolutional Neural Networks - **CNN** ou ConvNet) sont une catégorie de réseaux de neurones profonds (Deep Learning) spécialisés dans l'analyse de données visuelles comme les images et les vidéos. Inspirés par le cortex visuel humain, ils apprennent automatiquement une hiérarchie de caractéristiques (features) à partir des données brutes. Ils sont très performants pour des tâches telles que la classification d'images, la détection d'objets, et la segmentation.

#### Architecture Typique

Une architecture CNN standard se compose généralement des couches suivantes, organisées séquentiellement :

1. **Couche d'Entrée (Input Layer) :** Reçoit les données brutes (ex: image sous forme de tenseur `hauteur x largeur x canaux`).
2. **Blocs Convolutifs (répétés N fois) :**
   * **Couche Convolutive (Convolutional Layer) :** Extrait des caractéristiques locales à l'aide de filtres.
   * **Couche d'Activation (Activation Layer) :** Introduit la non-linéarité (souvent **ReLU**).
   * **Couche de Pooling (Pooling Layer) :** Réduit la dimension spatiale (sous-échantillonnage).
3. **Couche d'Aplatissement (Flattening Layer) :** Convertit les cartes de caractéristiques multi-dimensionnelles en un vecteur 1D.
4. **Couches Entièrement Connectées (Fully Connected Layers - FCL) :** Une ou plusieurs couches denses (type MLP) qui réalisent la classification/régression finale à partir des caractéristiques extraites. La dernière couche utilise une fonction d'activation adaptée à la tâche (ex: **Softmax** pour la classification multi-classes).

#### Hyperparamètres

Ce sont les paramètres configurés *avant* l'entraînement :

* **Pour les Couches Convolutives :**
  * Nombre de filtres
  * Taille des filtres (*kernel size*, ex: 3x3)
  * Pas (*stride*)
  * Remplissage (*padding*)
* **Pour les Couches de Pooling :**
  * Type de pooling (**Max**, **Average**)
  * Taille de la fenêtre de pooling (ex: 2x2)
  * Pas (*stride*)
* **Pour les Couches Entièrement Connectées :**
  * Nombre de neurones par couche
  * Fonction d'activation
* **Paramètres Généraux :**
  * Taux d'apprentissage (*learning rate*)
  * Fonction de perte (*loss function*)
  * Optimiseur (Adam, SGD...)
  * Nombre d'époques (*epochs*)
  * Taille des lots (*batch size*)
  * Techniques de régularisation (ex: *Dropout rate*)

### 2. Couche Convolutive et Filtre de Convolution

#### Principe de Fonctionnement

Une **couche convolutive** applique une opération de **convolution** : un petit filtre (ou noyau, *kernel*) glisse sur l'entrée (image ou feature map précédente). À chaque position, elle calcule le produit scalaire entre les poids du filtre et la zone correspondante de l'entrée. Le résultat forme un point dans la carte de caractéristiques de sortie (**Feature Map**). Le point clé est le **partage de poids** : le même filtre est utilisé partout, permettant de détecter une caractéristique indépendamment de sa position.

#### Filtre de Convolution

Un **filtre** est une petite matrice de poids (ex: 3x3, 5x5) dont les valeurs sont apprises durant l'entraînement. Chaque filtre est spécialisé dans la détection d'un motif spécifique (bords, coins, textures, etc.). Une couche convolutive utilise typiquement plusieurs filtres en parallèle pour capturer une diversité de caractéristiques.

### 3. Fonction d'Activation (ReLU)

#### Fonction Utilisée

La fonction d'activation prédominante dans les couches cachées des CNN est **ReLU (Rectified Linear Unit)**.
Sa définition est : $f(x) = max(0, x)$.

#### Pourquoi ReLU est Adaptée

1. **Efficacité Computationnelle :** Calcul très rapide comparé à `sigmoid` ou `tanh`.
2. **Atténuation du Problème de Disparition du Gradient :** Sa dérivée constante (1 pour $x > 0$) facilite la propagation du gradient lors de la rétropropagation, accélérant l'apprentissage.
3. **Sparsité :** En mettant à zéro les activations négatives, ReLU peut rendre les activations du réseau plus éparses.

### 4. Feature Map (Carte de Caractéristiques)

Une **Feature Map** (ou carte de caractéristiques) est le résultat produit par l'application d'un filtre de convolution suivi d'une fonction d'activation sur une entrée. C'est une matrice 2D indiquant où et avec quelle intensité la caractéristique spécifique détectée par le filtre est présente dans l'entrée. Une couche convolutive génère autant de feature maps qu'elle a de filtres.

### 5. Couche de Pooling

#### Principe de Fonctionnement

Une **couche de pooling** réduit la dimension spatiale (largeur, hauteur) des feature maps. Elle opère sur de petites fenêtres (ex: 2x2) et remplace chaque fenêtre par une seule valeur agrégée.
Objectifs :

* Réduire le nombre de paramètres et la complexité computationnelle.
* Contrôler le surapprentissage (*overfitting*).
* Introduire une certaine invariance aux petites translations.

#### Opérations de Pooling (Exemples)

1. **Max Pooling :** Conserve la valeur maximale de chaque fenêtre. Très courant, préserve les caractéristiques saillantes.
2. **Average Pooling :** Calcule la moyenne des valeurs de chaque fenêtre. A un effet lissant.

### 6. Couche Entièrement Connectée (Fully Connected Layer - FCL)

#### Fonctionnement

Après les couches convolutives et de pooling, les feature maps finales (représentant des caractéristiques de haut niveau) sont **aplaties** en un unique **vecteur 1D**. Ce vecteur est ensuite passé à une ou plusieurs **couches entièrement connectées (FCL)**. Dans une FCL, chaque neurone est connecté à toutes les activations de la couche précédente. Ces couches apprennent des combinaisons non-linéaires des caractéristiques extraites pour réaliser la tâche finale (ex: classification via **Softmax**).

#### Ce qu'elle Reçoit

La première FCL reçoit le **vecteur 1D** résultant de l'aplatissement des dernières cartes de caractéristiques issues des couches convolutives/pooling.

### 7. Pourquoi Préférer les CNN aux Réseaux Denses pour les Images ?

Les CNN surpassent les réseaux denses (MLP) pour les tâches d'images pour ces raisons :

1. **Exploitation de la Structure Spatiale :** Les convolutions opèrent localement et préservent l'information spatiale 2D des images, contrairement aux réseaux denses qui traitent les pixels comme indépendants.
2. **Partage de Poids :** L'utilisation répétée des mêmes filtres sur l'image réduit considérablement le nombre de paramètres par rapport à une connexion dense, limitant le surapprentissage et améliorant l'efficacité.
3. **Hiérarchie des Caractéristiques :** Les CNN apprennent naturellement des caractéristiques de complexité croissante, des bords simples aux objets complexes, à travers les couches successives.
4. **Invariance (partielle) aux Translations :** Le pooling et le partage de poids rendent le réseau moins sensible à la position exacte des objets dans l'image.

# Données et leur analyse

# Algorithmes utilisés

# Conclusion
