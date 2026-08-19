# Rock-Paper-Scissors : pièges d'implémentation

Notes techniques sur `RPS_game.py` (boilerplate non modifiable), distinctes du cadrage
conceptuel. Directement liées à l'objectif de comparer plusieurs stratégies interchangeables
via un registre de fonctions : ces pièges casseraient silencieusement un tel benchmark si on
ne les anticipe pas.

## État mutable partagé entre appels (mutable default arguments)

`quincy(prev_play, counter=[0])`, `mrugesh(..., opponent_history=[])`, `abbey(...,
opponent_history=[], play_order=[{...}])` utilisent des listes/dicts par défaut, créés **une
seule fois** à la définition de la fonction, donc **persistants entre tous les appels**, y compris
entre matchs différents si le process n'est pas redémarré. Si on enchaîne plusieurs `play()` dans
le même notebook Marimo pour comparer nos stratégies, l'historique d'un match précédent contamine
le suivant.

## `abbey` particulièrement sensible

Sa table `play_order` accumule sur *tout* l'historique cumulé du process. Comparer notre stratégie
A puis notre stratégie B contre `abbey` dans le même run donnerait un résultat invalide pour B
(elle hériterait de la mémoire du match de A). Il faut redémarrer le kernel/process entre deux
comparaisons de stratégies, ou réinitialiser explicitement l'état des bots avant chaque nouveau
match.

## Cas `prev_play == ""` au premier coup

Kris et Abbey gèrent explicitement ce cas (`if prev_opponent_play == '': prev_opponent_play =
"R"`), notre propre `player()` doit faire de même dès le premier appel, sinon KeyError ou
comportement indéfini au tour 1.

## `mrugesh` n'est pas déterministe entre deux processus

Piège le plus discret des quatre, et le seul qui rende un résultat non reproductible :

```python
most_frequent = max(set(last_ten), key=last_ten.count)
```

En cas d'**égalité de fréquence** entre deux coups sur les dix derniers, `max` renvoie le premier
rencontré dans l'ordre d'itération du `set`. Or cet ordre dépend du hachage des chaînes, donc de
`PYTHONHASHSEED`, qui est aléatoire à chaque démarrage de Python.

Mesuré sur la stratégie retenue, mêmes 1000 coups :

| `PYTHONHASHSEED` | quincy | abbey | kris | mrugesh |
|---|---|---|---|---|
| 0 | 99,90 | 67,07 | 75,70 | **84,62** |
| 1 | 99,90 | 67,07 | 75,70 | **84,21** |
| 7 | 99,90 | 67,07 | 75,70 | **84,51** |
| 42 | 99,90 | 67,07 | 75,70 | **83,82** |

Les trois autres bots sont parfaitement déterministes : seul mrugesh bouge, dans une plage
d'environ 0,8 point. L'écart est plus marqué sur des stratégies faibles (`self_model` contre
mrugesh varie de 63,8 à 75,4 % selon le processus).

**Conséquences pratiques :**

- Fixer une graine Python (`random.seed`) ne change rien : ce n'est pas le module `random` qui est
  en cause, mais le hachage des chaînes du `set`.
- Un chiffre mesuré contre mrugesh doit être donné comme **approximatif** (« ~84 % »), jamais à la
  décimale près, sauf à préciser le `PYTHONHASHSEED` utilisé.
- Pour reproduire un résultat à l'identique, lancer avec `PYTHONHASHSEED=0 python ...`.
- Un test unitaire ne doit pas asserter une valeur exacte contre mrugesh.

## Nom de paramètre trompeur

Le premier paramètre de chaque bot (`prev_play` ou `prev_opponent_play`) désigne **notre** dernier
coup à nous, pas celui du bot, `play()` appelle `player1(p2_prev_play)` et
`player2(p1_prev_play)`. Source de confusion si on ne relit pas attentivement qui reçoit quoi.

## `play()` mesure le taux de victoire de Player 1 uniquement

`results['p1'] / games_won * 100`, notre fonction doit être passée en première position
(`play(player, bot, 1000)`) pour que le taux rapporté soit bien le nôtre.
