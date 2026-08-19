# Linear Regression Health Costs Calculator — énoncé freeCodeCamp

Traduction française de l'énoncé officiel du projet, conservée telle quelle comme référence :
c'est le cahier des charges imposé, pas un document de travail.

Source : https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/linear-regression-health-costs-calculator

## Table des matières

- [Objectif](#objectif)
- [Environnement de travail](#environnement-de-travail)
- [Préparation des données](#préparation-des-données)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Évaluation](#évaluation)
- [Critères d'acceptation](#critères-dacceptation)

## Objectif

Prédire des **coûts de santé** à l'aide d'un algorithme de **régression**.

Le jeu de données fourni contient des informations sur différentes personnes, dont leurs coûts de
santé. Il faut utiliser ces données pour prédire les coûts de santé à partir de données nouvelles.

## Environnement de travail

Le projet se fait sur **Google Colaboratory**. Il faut créer une copie du notebook, soit sur son
propre compte, soit en local. Une fois le projet terminé et le test (inclus dans le notebook)
passé, on soumet le lien du projet. Pour un lien Google Colaboratory, penser à **activer le
partage du lien pour « toute personne disposant du lien »**.

Le contenu pédagogique interactif du cursus machine learning est encore en développement chez
freeCodeCamp. En attendant, on peut suivre les défis vidéo de cette certification, et il faut
chercher des ressources d'apprentissage complémentaires — comme on le ferait sur un vrai projet.

## Préparation des données

Les **deux premières cellules** du notebook importent les bibliothèques et les données.

- **Convertir les données catégorielles en nombres.**
- Utiliser **80 % des données** comme `train_dataset` et **20 %** comme `test_dataset`.
- **Retirer** (`pop`) la colonne **`expenses`** de ces jeux de données pour créer deux nouveaux
  jeux, `train_labels` et `test_labels`. Ce sont ces étiquettes qu'il faut utiliser pour
  entraîner le modèle.

## Entraînement du modèle

Créer un modèle et l'entraîner avec `train_dataset`.

## Évaluation

Exécuter la dernière cellule du notebook pour contrôler le modèle. Cette cellule utilise le
`test_dataset` — jamais vu par le modèle — pour vérifier sa capacité de généralisation.

Pour valider le défi, **`model.evaluate` doit renvoyer une erreur absolue moyenne (MAE)
inférieure à 3500**, c'est-à-dire que le modèle prédit les coûts de santé à moins de 3500 dollars
près.

La dernière cellule prédit également les dépenses à partir du `test_dataset` et trace les
résultats.

## Critères d'acceptation

| Critère | Valeur imposée |
|---|---|
| Type de modèle | Régression |
| Répartition des données | 80 % entraînement / 20 % test |
| Colonne cible | `expenses`, retirée en `train_labels` / `test_labels` |
| Données catégorielles | À convertir en nombres |
| Seuil de réussite | MAE < 3500 (via `model.evaluate`) |
| Livrable | Lien du notebook, partage activé |
