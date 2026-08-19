# Neural Network SMS Text Classifier — énoncé freeCodeCamp

Traduction française de l'énoncé officiel du projet, conservée telle quelle comme référence :
c'est le cahier des charges imposé, pas un document de travail.

Source : https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier

## Table des matières

- [Objectif](#objectif)
- [Environnement de travail](#environnement-de-travail)
- [La fonction `predict_message`](#la-fonction-predict_message)
- [Le jeu de données](#le-jeu-de-données)
- [Organisation du notebook](#organisation-du-notebook)
- [Critères d'acceptation](#critères-dacceptation)

## Objectif

Créer un modèle de machine learning qui classe des SMS en **« ham »** ou **« spam »** :

- un message **« ham »** est un message normal, envoyé par un ami ;
- un message **« spam »** est une publicité ou un message envoyé par une entreprise.

## Environnement de travail

Le projet se fait sur **Google Colaboratory**. Il faut créer une copie du notebook, soit sur son
propre compte, soit en local. Une fois le projet terminé et le test (inclus dans le notebook)
passé, on soumet le lien du projet. Pour un lien Google Colaboratory, penser à **activer le
partage du lien pour « toute personne disposant du lien »**.

Le contenu pédagogique interactif du cursus machine learning est encore en développement chez
freeCodeCamp. En attendant, on peut suivre les défis vidéo de cette certification, et il faut
chercher des ressources d'apprentissage complémentaires — comme on le ferait sur un vrai projet.

## La fonction `predict_message`

Créer une fonction nommée **`predict_message`** qui prend en argument une chaîne de caractères
(le message) et renvoie une **liste** :

- le **premier élément** est un nombre entre zéro et un, indiquant la vraisemblance de
  « ham » (**0**) ou de « spam » (**1**) ;
- le **second élément** est le mot **`"ham"`** ou **`"spam"`**, selon celui qui est le plus
  probable.

## Le jeu de données

Le projet utilise le jeu de données **SMS Spam Collection**. Il est **déjà réparti** en données
d'entraînement (*train*) et données de test (*test*).

## Organisation du notebook

Les **deux premières cellules** importent les bibliothèques et les données. La **dernière
cellule** teste le modèle et la fonction. Tout le code est à ajouter **entre ces cellules**.

## Critères d'acceptation

| Critère | Valeur imposée |
|---|---|
| Tâche | Classification binaire ham / spam |
| Jeu de données | SMS Spam Collection (train/test déjà séparés) |
| Fonction à écrire | `predict_message(message)` |
| Sortie | `[probabilité (0 = ham, 1 = spam), "ham" ou "spam"]` |
| Livrable | Lien du notebook, partage activé |
