# Book Recommendation Engine using KNN — énoncé freeCodeCamp

Traduction française de l'énoncé officiel du projet, conservée telle quelle comme référence :
c'est le cahier des charges imposé, pas un document de travail.

Source : https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/book-recommendation-engine-using-knn

## Table des matières

- [Objectif](#objectif)
- [Environnement de travail](#environnement-de-travail)
- [Le jeu de données](#le-jeu-de-données)
- [Le modèle](#le-modèle)
- [La fonction `get_recommends`](#la-fonction-get_recommends)
- [Filtrage du jeu de données](#filtrage-du-jeu-de-données)
- [Organisation du notebook](#organisation-du-notebook)
- [Critères d'acceptation](#critères-dacceptation)

## Objectif

Créer un algorithme de recommandation de livres à l'aide des **k plus proches voisins**
(K-Nearest Neighbors).

## Environnement de travail

Le projet se fait sur **Google Colaboratory**. Il faut créer une copie du notebook, soit sur son
propre compte, soit en local. Une fois le projet terminé et le test (inclus dans le notebook)
passé, on soumet le lien du projet. Pour un lien Google Colaboratory, penser à **activer le
partage du lien pour « toute personne disposant du lien »**.

Le contenu pédagogique interactif du cursus machine learning est encore en développement chez
freeCodeCamp. En attendant, on peut suivre les défis vidéo de cette certification, et il faut
chercher des ressources d'apprentissage complémentaires — comme on le ferait sur un vrai projet.

## Le jeu de données

Le projet utilise le jeu de données **Book-Crossings**, qui contient **1,1 million de notes**
(échelle de 1 à 10) portant sur **270 000 livres**, données par **90 000 utilisateurs**.

Le jeu de données est **déjà importé dans le notebook** : aucun téléchargement supplémentaire
n'est nécessaire.

## Le modèle

Utiliser **`NearestNeighbors` de `sklearn.neighbors`** pour développer un modèle qui montre les
livres similaires à un livre donné. L'algorithme des plus proches voisins mesure la distance pour
déterminer la « proximité » entre instances.

## La fonction `get_recommends`

Créer une fonction nommée **`get_recommends`** qui prend en argument un titre de livre (présent
dans le jeu de données) et renvoie une liste de **5 livres similaires accompagnés de leur
distance** au livre passé en argument.

Ce code :

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```

doit renvoyer :

```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301],
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479],
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

Structure de la valeur de retour — c'est une **liste** :

- le **premier élément** est le titre du livre passé à la fonction ;
- le **second élément** est une liste de **cinq listes** ; chacune de ces cinq listes contient un
  livre recommandé et la distance entre ce livre recommandé et le livre passé à la fonction.

## Filtrage du jeu de données

En traçant le jeu de données (optionnel), on constate que la plupart des livres sont rarement
notés. Pour garantir la **significativité statistique**, retirer du jeu de données :

- les **utilisateurs ayant moins de 200 notes** ;
- les **livres ayant moins de 100 notes**.

## Organisation du notebook

Les **trois premières cellules** importent les bibliothèques éventuellement nécessaires ainsi que
les données à utiliser. La **dernière cellule** sert aux tests. Tout le code est à écrire **entre
ces cellules**.

## Critères d'acceptation

| Critère | Valeur imposée |
|---|---|
| Algorithme | `NearestNeighbors` (`sklearn.neighbors`) |
| Jeu de données | Book-Crossings (déjà importé dans le notebook) |
| Fonction à écrire | `get_recommends(titre)` |
| Sortie | `[titre, [[livre, distance] × 5]]` |
| Seuil utilisateurs | ≥ 200 notes |
| Seuil livres | ≥ 100 notes |
| Livrable | Lien du notebook, partage activé |
