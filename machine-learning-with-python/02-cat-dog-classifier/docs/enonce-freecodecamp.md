# Cat and Dog Image Classifier — énoncé freeCodeCamp

Traduction française de l'énoncé officiel du projet, conservée telle quelle comme référence :
c'est le cahier des charges imposé, pas un document de travail.

Source : https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier

## Table des matières

- [Objectif](#objectif)
- [Environnement de travail](#environnement-de-travail)
- [Structure du jeu de données](#structure-du-jeu-de-données)
- [Consignes cellule par cellule](#consignes-cellule-par-cellule)
  - [Cellule 3 — générateurs d'images](#cellule-3--générateurs-dimages)
  - [Cellule 4 — `plotImages`](#cellule-4--plotimages)
  - [Cellule 5 — augmentation de données](#cellule-5--augmentation-de-données)
  - [Cellule 6 — visualisation des variations](#cellule-6--visualisation-des-variations)
  - [Cellule 7 — le modèle](#cellule-7--le-modèle)
  - [Cellule 8 — entraînement](#cellule-8--entraînement)
  - [Cellule 9 — courbes](#cellule-9--courbes)
  - [Cellule 10 — prédictions](#cellule-10--prédictions)
  - [Cellule 11 — vérification finale](#cellule-11--vérification-finale)
- [Critères d'acceptation](#critères-dacceptation)

## Objectif

Compléter le code permettant de classifier des images de chiens et de chats. Il faut utiliser
**TensorFlow 2.0 et Keras** pour créer un **réseau de neurones convolutif** (CNN) qui classe
correctement les images de chats et de chiens **au moins 63 % du temps**. (Bonus si l'on atteint
70 % de justesse.)

Une partie du code est fournie, le reste est à compléter. Chaque cellule de texte du notebook
indique ce qu'il y a à faire dans la cellule de code qui suit.

- la première cellule de code importe les bibliothèques nécessaires ;
- la deuxième télécharge les données et fixe les variables clés ;
- la troisième est le premier endroit où écrire son propre code.

## Environnement de travail

Le projet se fait sur **Google Colaboratory**. Il faut créer une copie du notebook, soit sur son
propre compte, soit en local. Une fois le projet terminé et le test passé, on soumet le lien du
projet. Pour un lien Google Colaboratory, penser à **activer le partage du lien pour « toute
personne disposant du lien »**.

Le contenu pédagogique interactif du cursus machine learning est encore en développement chez
freeCodeCamp. En attendant, on peut suivre les défis vidéo de cette certification, et il faut
chercher des ressources d'apprentissage complémentaires — comme on le ferait sur un vrai projet.

## Structure du jeu de données

Le répertoire `test` n'a **pas de sous-répertoires** et ses images ne sont **pas étiquetées** :

```
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```

Il est possible d'ajuster le nombre d'epochs et la taille de batch, mais ce n'est pas exigé.

Les consignes ci-dessous correspondent à des numéros de cellule précis, indiqués par un
commentaire en haut de chaque cellule (par exemple `# 3`).

## Consignes cellule par cellule

### Cellule 3 — générateurs d'images

Renseigner correctement chacune des variables de la cellule (elles ne doivent plus valoir `None`).

Créer un générateur d'images pour chacun des trois jeux de données (`train`, `validation`,
`test`). Utiliser `ImageDataGenerator` pour lire/décoder les images et les convertir en tenseurs
de nombres flottants. Utiliser l'argument `rescale` (**et aucun autre pour l'instant**) pour
ramener les valeurs des tenseurs de l'intervalle 0–255 à l'intervalle 0–1.

Pour les variables `*_data_gen`, utiliser la méthode `flow_from_directory` en lui passant la
taille de batch, le répertoire, la taille cible (`(IMG_HEIGHT, IMG_WIDTH)`), le mode de classe, et
tout autre argument nécessaire.

`test_data_gen` est le plus délicat : lui passer **`shuffle=False`** dans `flow_from_directory`,
afin que les prédictions finales restent dans l'ordre attendu par le test. Pour `test_data_gen`,
il est également utile d'observer la structure du répertoire.

Sortie attendue après exécution :

```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 50 images belonging to 1 class.
```

### Cellule 4 — `plotImages`

La fonction `plotImages` sert à afficher des images à plusieurs reprises. Elle prend un tableau
d'images et une liste de probabilités (cette dernière étant optionnelle). **Ce code est fourni.**
Si la variable `train_data_gen` a été correctement créée, l'exécution de cette cellule affiche
cinq images d'entraînement prises au hasard.

### Cellule 5 — augmentation de données

Recréer `train_image_generator` avec `ImageDataGenerator`.

Le nombre d'exemples d'entraînement étant faible, il y a un risque de surapprentissage
(*overfitting*). Un moyen d'y remédier est de créer davantage de données d'entraînement à partir
des exemples existants, au moyen de transformations aléatoires.

Ajouter **4 à 6 transformations aléatoires** en arguments d'`ImageDataGenerator`. Veiller à
appliquer le même `rescale` que précédemment.

### Cellule 6 — visualisation des variations

Rien à faire dans cette cellule. `train_data_gen` y est recréé comme précédemment, mais avec le
nouveau `train_image_generator`. Une même image est ensuite affichée cinq fois avec des variations
différentes.

### Cellule 7 — le modèle

Créer un modèle de réseau de neurones qui produit des **probabilités de classe**. Il doit utiliser
le modèle `Sequential` de Keras. Il comportera probablement un empilement de couches `Conv2D` et
`MaxPooling2D`, puis une couche entièrement connectée au sommet, activée par une fonction
d'activation **ReLU**.

Compiler le modèle en passant les arguments qui fixent l'optimiseur et la fonction de perte.
Passer également `metrics=['accuracy']` pour voir la justesse en entraînement et en validation à
chaque epoch.

### Cellule 8 — entraînement

Utiliser la méthode `fit` du modèle pour entraîner le réseau. Passer les arguments `x`,
`steps_per_epoch`, `epochs`, `validation_data` et `validation_steps`.

### Cellule 9 — courbes

Exécuter cette cellule pour visualiser la justesse et la perte du modèle.

### Cellule 10 — prédictions

Utiliser le modèle pour prédire si une image inédite est un chat ou un chien.

Obtenir la probabilité que chaque image de test (issue de `test_data_gen`) soit un chien ou un
chat. `probabilities` doit être une liste d'entiers.

Appeler `plotImages` en lui passant les images de test et les probabilités correspondantes.

Après exécution, les 50 images de test doivent s'afficher avec une étiquette indiquant le
pourcentage de « certitude » que l'image soit un chat ou un chien. La justesse correspondra à
celle affichée sur le graphique précédent. Davantage d'images d'entraînement permettraient une
meilleure justesse.

### Cellule 11 — vérification finale

Exécuter cette dernière cellule pour savoir si le défi est validé ou s'il faut continuer à
chercher.

## Critères d'acceptation

| Critère | Valeur imposée |
|---|---|
| Justesse minimale | 63 % (70 % en bonus) |
| Bibliothèques | TensorFlow 2.0 + Keras |
| Type de modèle | CNN, `Sequential` (Conv2D / MaxPooling2D + couche dense ReLU) |
| Jeu d'entraînement | 2000 images, 2 classes |
| Jeu de validation | 1000 images, 2 classes |
| Jeu de test | 50 images, non étiquetées, `shuffle=False` |
| Transformations aléatoires (cellule 5) | 4 à 6 |
| Livrable | Lien du notebook, partage activé |
