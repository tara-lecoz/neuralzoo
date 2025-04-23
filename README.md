# Contexte du projet

# Veille

---

## I. Perceptron Multicouches (MLP) : Synthèse

### 1. Architecture du Perceptron Multicouche (PMC)

Le Perceptron Multicouche (Multilayer Perceptron - MLP) est un type de réseau de neurones artificiels structuré en plusieurs couches de neurones :

* **Couche d'entrée** : elle reçoit les données d'entrée. Chaque neurone de cette couche correspond à une feature de l'entrée.
* **Couches cachées** : situées entre l'entrée et la sortie, elles permettent au réseau de modéliser des relations complexes. On parle souvent de couches **denses** lorsque chaque neurone est connecté à tous les neurones de la couche précédente.
* **Couche de sortie** : elle donne le résultat final du modèle. Le nombre de neurones dépend du type de tâche (classification binaire, multi-classes, régression).

#### Hyperparamètres typiques :

* Nombre de couches cachées
* Nombre de neurones par couche
* Fonction(s) d'activation
* Taux d'apprentissage (learning rate)
* Méthode d'optimisation (Adam, SGD, etc.)
* Nombre d'épochs
* Taille du batch

### 2. Choix de l'architecture selon la problématique

* **Classification** :
  * Couche de sortie avec une activation **sigmoïde** (binaire) ou **softmax** (multi-classes).
  * Fonction de perte : entropie croisée (cross-entropy).
* **Régression** :
  * Couche de sortie avec une activation **linéaire**.
  * Fonction de perte : erreur quadratique moyenne (MSE).

Le nombre de couches cachées et de neurones dépend de la complexité du problème, du volume de données, et de la capacité de généralisation souhaitée.

### 3. Définitions de termes clés

* **Fonction d'activation** : transforme la sortie d'un neurone de façon non-linéaire. Permet de modéliser des fonctions complexes.
* **Propagation (forward propagation)** : phase où les données d'entrée traversent le réseau jusqu'à la sortie.
* **Rétropropagation (backpropagation)** : méthode d'entraînement qui ajuste les poids en fonction de l'erreur.
* **Loss function (fonction de coût)** : mesure l'écart entre la sortie prédite et la vraie valeur.
* **Descente de gradient** : algorithme d'optimisation qui ajuste les poids pour minimiser la perte.
* **Vanishing gradients** : phénomène où les gradients deviennent très petits, ralentissant voire bloquant l'apprentissage, surtout dans les réseaux profonds.

### 4. Fonction d’activation : Définition et exemples

Une **fonction d’activation** introduit de la non-linéarité dans le modèle. Sans elle, le réseau ne pourrait modéliser que des relations linéaires.

Exemples :

* **ReLU (Rectified Linear Unit)** : `f(x) = max(0, x)`
* **Sigmoïde** : `f(x) = 1 / (1 + exp(-x))`
* **Tanh** : `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
* **Softmax** : utilisée pour la classification multi-classes, donne une distribution de probabilité.

### 5. Epochs, Iterations et Batch size

* **Epoch** : une passe complète sur l'ensemble des données d'entraînement.
* **Batch size** : nombre d'échantillons traités avant une mise à jour des poids.
* **Iteration** : une mise à jour des poids. Nombre d’itérations = nombre d’échantillons / batch size * nombre d’époques.

### 6. Taux d’apprentissage (learning rate)

Le **learning rate** détermine l’amplitude des mises à jour des poids à chaque itération.

* Trop bas : apprentissage lent, risque de stagnation.
* Trop élevé : apprentissage instable, risque de divergence.

### 7. Batch Normalization

La **batch normalization** est une technique qui consiste à normaliser les activations d'une couche en les centrant et réduisant, puis en les réajustant.

**Avantages** :

* Accélère l'entraînement
* Stabilise la propagation des gradients
* Permet des learning rates plus élevés

### 8. Algorithme d’optimisation Adam

**Adam (Adaptive Moment Estimation)** est un algorithme d’optimisation combinant les avantages de l’optimisation par **momentum** et de l’**adaptation du learning rate** pour chaque paramètre.

**Avantages** :

* Convergence rapide
* Moins sensible au choix du learning rate
* Très utilisé pour l’apprentissage profond

### 9. Définition simple du Perceptron Multicouche

Un **Perceptron Multicouche** est un réseau de neurones composé d’une couche d’entrée, de plusieurs couches cachées (denses), et d’une couche de sortie. Il est capable de modéliser des relations non-linéaires complexes et peut être utilisé pour des tâches de classification ou de régression.

ReLU permet d’introduire de la non-linéarité tout en limitant le problème du gradient qui disparaît. D’autres fonctions comme `tanh` ou `sigmoid` peuvent également être utilisées selon les cas.

#### 3. Couche Entièrement Connectée (Fully Connected Layer - FCL)

Le MLP repose **entièrement** sur des couches entièrement connectées. Chaque neurone d’une couche est connecté à **tous** les neurones de la couche précédente. Cela permet une  **forte capacité d’approximation** , mais  **augmente considérablement le nombre de paramètres** , surtout avec des images de grande taille.

---

## II. Réseaux de Neurones Convolutifs (CNN) : Synthèse

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

---

## Données et leur analyse

# Algorithmes utilisés

# Conclusion
