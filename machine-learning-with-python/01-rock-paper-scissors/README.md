# 01 : Rock Paper Scissors

Premier des 5 projets de la certification *Machine Learning with Python*. Écrire une fonction
`player()` qui bat 4 bots à au moins 60 % sur 1000 coups.

## Par où entrer

**Pour comprendre ce qui est demandé** → [docs/freecodecamp/enonce-freecodecamp.md](docs/freecodecamp/enonce-freecodecamp.md)
(traduction de l'énoncé officiel) puis [docs/freecodecamp/implementation-notes.md](docs/freecodecamp/implementation-notes.md)
(pièges de `RPS_game.py` : à lire **avant** de coder, ils cassent silencieusement un benchmark).

**Pour comprendre le raisonnement** → le notebook `notebook.py`, qui déroule la démarche acte
par acte : cadrage du problème, choix du modèle, mesure, et ce que la mesure a démenti.

Le cadrage théorique complet (chaîne de Markov d'ordre variable, comptage multinomial, lissage
bayésien) et les décisions d'architecture qui en découlent sont documentés séparément, hors de
ce dépôt : ce projet sert de terrain pour roder une méthode, et cette exploration n'est pas
nécessaire pour comprendre le livrable.

## Fichiers

| Fichier | Rôle |
|---|---|
| `RPS_game.py` | Boilerplate freeCodeCamp : **ne pas modifier** (vérifié identique à l'upstream) |
| `RPS.py` | Le livrable : `player()`, couche mince au-dessus des modules ci-dessous |
| `strategies.py` | Registre : `markov_order_0` à `3`, `uniform`, `self_model_1`, `self_model_2`, `mixture` |
| `decision.py` | Distribution prédite → coup joué (`best_response`, `sample`) |
| `metrics.py` | Brier, log-loss, IC de Wilson, calibration |
| `harness.py` | Banc de comparaison, avec isolation de l'état des bots |
| `main.py` | Bac à sable de développement |
| `test_module.py` | Tests officiels freeCodeCamp (seuil 60 % contre chacun des 4 bots), fourni par le boilerplate et versionné avec lui |
| `test_units.py` | Tests unitaires des modules : contrat, estimation, isolation d'état, métriques |
| `notebook.py` | Exploration Marimo : orchestration et visualisation uniquement |

Structure assumée : modules plats, pas de `src/`, aucune duplication entre le livrable et l'atelier.

Starter récupéré depuis
[freeCodeCamp/boilerplate-rock-paper-scissors](https://github.com/freeCodeCamp/boilerplate-rock-paper-scissors).

```bash
python -m unittest discover      # tous les tests (officiels + unitaires)
python -m unittest test_units    # tests unitaires seuls (rapide)
python -m unittest test_module   # les 4 tests officiels
python harness.py                # comparaison de toutes les stratégies
marimo edit --sandbox notebook.py   # le récit complet, avec graphes
```

**Commencer par le notebook** : il raconte le cheminement en quatre actes (plancher,
mémoire, le plafond d'abbey, le changement de question) avec les graphes de chaque étape,
la calibration et un banc interactif.

Notebook Marimo sandboxed, dépendances déclarées en en-tête PEP 723.

## État : les 4 tests passent

| Bot | Taux de victoire | Seuil 60 % |
|---|---|---|
| quincy | 99,9 % | atteint |
| abbey | 67,1 % | atteint |
| kris | 75,7 % | atteint |
| mrugesh | ~84 % | atteint |

Les trois premiers chiffres sont **exactement reproductibles**. Celui de mrugesh ne l'est pas, et
la raison mérite d'être connue : son code fait `max(set(last_ten), key=last_ten.count)`, et en cas
d'égalité de fréquence le vainqueur dépend de l'ordre d'itération du `set`, donc de
`PYTHONHASHSEED`, tiré au hasard à chaque démarrage de Python.

Mesuré sur quatre processus : 84,62 / 84,51 / 84,21 / 83,82 %. Une plage d'environ 0,8 point,
toujours très au-dessus du seuil. Donner « 83,8 % » serait présenter une réalisation particulière
comme le résultat. Détail dans
[implementation-notes.md](docs/freecodecamp/implementation-notes.md).

Pour reproduire un chiffre à l'identique : `PYTHONHASHSEED=0 python -m unittest test_module`.

**Ce que la mesure a appris.** La question du cadrage était « quel ordre de chaîne de Markov ? ».
La réponse mesurée est que **ce n'était pas la bonne question pour abbey**. La mémoire l'aide
beaucoup (+34,6 points entre l'ordre 0 et l'ordre 2, la plus forte progression des 4 bots), mais
elle s'arrête à 0,3 point du seuil et aucun ordre testé de 1 à 5 ne franchit 60 % de façon fiable
(à l'ordre 2, l'IC de Wilson vaut [56,1 ; 63,2], il contient encore 60 %). D'où `self_model_1`, qui
modélise ma propre séquence à l'ordre 1 (83,3 % contre abbey, mais ~50 % ailleurs), puis
`mixture`, qui sélectionne l'expert le plus performant au score de Brier sur une fenêtre
glissante.

L'ordre compte énormément ici, au point que le registre expose les deux variantes séparément :
`self_model_1` fait 83,3 % contre abbey, `self_model_2` seulement 49,5 %.

C'est ce qui justifie le contrat retenu : une stratégie renvoie une distribution, pas un
coup. La mixture serait impossible si les stratégies renvoyaient un coup : elle a besoin de la
distribution pour noter ses experts.

**Un arbitrage, pas une supériorité.** Comparée au meilleur expert seul, la mixture perd du
terrain là où Markov excellait :

| bot | markov ordre 2 | markov ordre 3 | mixture |
|---|---|---|---|
| quincy | 99,9 % | 99,9 % | 99,9 % |
| abbey | 59,7 % | 59,2 % | **67,1 %** |
| kris | 84,5 % | **91,4 %** | 75,7 % |
| mrugesh | ~84 % | ~84 % | ~84 %, jamais meilleur que l'ordre 2 |

Sur mrugesh, l'écart est trop petit pour être chiffré à la décimale (le bot n'est pas
déterministe, voir plus haut). Mesuré sur huit processus, la mixture reste **toujours en dessous
de Markov ordre 2**, de 0,01 à 0,51 point. Elle passe en revanche devant l'ordre 3 dans deux cas
sur huit : le classement entre ces deux-là n'est pas stable, contrairement à ce qu'un échantillon
plus petit laissait croire.

Elle ne gagne donc sur aucun des trois bots que Markov ordre 2 battait déjà.

En moyenne, Markov ordre 3 fait mieux. Mais le critère est le **minimum** sur les 4 bots : Markov
s'arrête à 59,7 % contre abbey, à 0,3 point du seuil. La mixture échange de la performance sur
kris contre le point qui manquait sur abbey. Une sélection plus prudente, ne quittant Markov que
lorsqu'il faiblit, garderait probablement les deux : c'est le prochain chantier.

**Le code des bots a été lu en dernier**, une fois la mesure faite, comme corrigé. Il a invalidé
ma typologie sans invalider mes résultats : je croyais opposer des bots à séquence fixe et des
bots réactifs, alors que **trois des quatre sont réactifs** (kris et mrugesh répondent à mes
coups, kris sans aucun état). Markov les bat quand même parce que leur réponse est une fonction
déterministe de mon jeu : leur séquence encode la mienne. Ce qui isole abbey n'est donc pas sa
réactivité mais son **état accumulé** (`play_order` compte mes paires de coups). Le vrai axe est
« réactif sans mémoire » contre « réactif avec état ». Détail dans le notebook, acte 3.

Chantiers ouverts : brancher davantage de métriques (regret cumulé,
exploitabilité), et remplacer la règle d'agrégation de la mixture (Follow-the-Leader) par Hedge,
qui offrirait une borne de regret là où la version actuelle perd 16 points sur kris.

URL de soumission : *à compléter une fois le projet validé.*
