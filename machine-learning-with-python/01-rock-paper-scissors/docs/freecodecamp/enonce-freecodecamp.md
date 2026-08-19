# Rock Paper Scissors : énoncé freeCodeCamp

Traduction française de l'énoncé officiel du projet, conservée telle quelle comme référence :
c'est le cahier des charges imposé, pas un document de travail. Les pièges techniques du
boilerplate sont relevés dans [implementation-notes.md](implementation-notes.md).

Source : https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors

## Table des matières

- [Objectif](#objectif)
- [Ce qui est fourni](#ce-qui-est-fourni)
- [La fonction `player`](#la-fonction-player)
- [Développement](#développement)
- [Tests](#tests)
- [Soumission](#soumission)
- [Critères d'acceptation](#critères-dacceptation)

## Objectif

Écrire un programme qui joue à Pierre-Papier-Ciseaux. Un programme qui choisit au hasard gagne
en général 50 % du temps. Pour valider ce défi, le programme doit affronter **quatre bots
différents** et gagner **au moins 60 % des parties dans chaque match**.

## Ce qui est fourni

Le projet démarre à partir du code de départ freeCodeCamp (fourni via Ona).

La partie pédagogique interactive du cursus machine learning est encore en développement chez
freeCodeCamp : il faut donc s'appuyer sur d'autres ressources pour apprendre ce qu'il faut pour
réussir ce défi.

## La fonction `player`

Le fichier `RPS.py` contient une fonction `player` à compléter :

- elle reçoit en argument une chaîne décrivant **le dernier coup de l'adversaire** (`"R"`, `"P"`
  ou `"S"`) ;
- elle doit renvoyer une chaîne représentant **le prochain coup à jouer** (`"R"`, `"P"` ou `"S"`).

Pour la première partie d'un match, la fonction reçoit une **chaîne vide** en argument, puisqu'il
n'y a pas de coup précédent.

La fonction d'exemple est définie avec deux arguments :

```python
player(prev_play, opponent_history = [])
```

Le second argument est **entièrement optionnel** : la fonction n'est jamais appelée avec. Il n'est
présent que parce que c'est le seul moyen de **conserver un état entre deux appels consécutifs**
de la fonction. Il n'est utile que si l'on souhaite garder trace de l'historique de l'adversaire.

**Indice de l'énoncé** : pour battre les quatre adversaires, le programme aura probablement besoin
de **plusieurs stratégies**, choisies selon la façon de jouer de l'adversaire.

## Développement

- **Ne pas modifier `RPS_game.py`.** Tout le code doit être écrit dans `RPS.py`.
- `main.py` sert aux essais pendant le développement ; il importe la fonction de jeu et les bots
  depuis `RPS_game.py`.

Pour tester, on lance une partie avec la fonction `play`, qui prend quatre arguments :

```python
play(player1, player2, num_games[, verbose])
```

- les deux joueurs qui s'affrontent (ce sont en réalité des fonctions) ;
- le nombre de parties à jouer dans le match ;
- un argument optionnel `verbose` : le passer à `True` affiche le détail de chaque partie.

Exemple : faire jouer `player` contre `quincy` sur 1000 parties en affichant les résultats de
chaque partie :

```python
play(player, quincy, 1000, verbose=True)
```

## Tests

Les tests unitaires du projet sont dans `test_module.py`. Ils sont importés dans `main.py` pour
plus de commodité : en décommentant la dernière ligne de `main.py`, les tests s'exécutent
automatiquement à chaque `python main.py` lancé dans la console.

## Soumission

Copier l'URL du projet et la soumettre à freeCodeCamp.

## Critères d'acceptation

| Critère | Valeur imposée |
|---|---|
| Adversaires à battre | 4 bots (`quincy`, `abbey`, `kris`, `mrugesh`) |
| Taux de victoire minimal | 60 % dans **chaque** match |
| Longueur d'un match | 1000 parties |
| Fichier à écrire | `RPS.py` (fonction `player`) |
| Fichier à ne pas modifier | `RPS_game.py` |
