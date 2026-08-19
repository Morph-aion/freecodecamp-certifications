# 03 : Book Recommendation Engine using KNN

Troisième des 5 projets de la certification *Machine Learning with Python*.
Recommander 5 livres similaires à un livre donné avec les **k plus proches
voisins** (`NearestNeighbors` de scikit-learn, distance cosinus), sur le jeu de
données Book-Crossings (1 149 780 notes, 271 379 livres, 105 283 utilisateurs ;
l'énoncé arrondit ce dernier chiffre à 90 000).

## Ce qui change par rapport au projet 02

Cat/Dog était un problème d'images : chaque observation est un tenseur, l'ordre
ne porte rien, un réseau apprend une fonction de classes. Ici, le problème est
**relationnel** : l'information utile n'est pas dans chaque note isolée mais
dans les **chevauchements de goûts** entre profils de notes. Le vocabulaire du
projet 01 (adversaire, non-stationnarité) ne s'applique pas davantage.

Le risque principal n'est pas le surapprentissage (il n'y a pas de paramètre à
surajuster dans k-NN, l'algorithme est paresseux) mais la **parcimonie** : le
jeu est si creux que sans filtrage, la plupart des livres n'ont pas assez de
notes pour que leur distance veuille dire quelque chose. Le projet est donc
presque entièrement un travail de **préparation de données** : le modèle tient
en trois lignes.

À quoi s'ajoute un risque qui ne se voit pas dans le code : celui de croire que
l'on mesure une similarité de goût alors que les trois quarts des notes ne sont
pas des jugements (voir plus bas). L'erreur ne fait échouer aucun test, elle
fausse seulement l'interprétation.

## Lancement

```bash
uv run --with marimo marimo edit --sandbox notebook.py            # notebook Marimo
uv run --with pandas --with scikit-learn python -m unittest test_units   # tests, moins de 15 s
```

Depuis la racine de la certification, le `Makefile` évite de retenir ces lignes :

```bash
make test PROJET=03-book-recommendation-knn   # la suite de ce projet
make lint                                     # ruff check + format --check
make fix                                      # corrige et formate
```

Les tests passent par `uv run` : le projet ne versionne pas d'environnement
Python, `uv` résout les dépendances à la volée. Le `.venv/` que Marimo crée en
local est gitignoré et ne fait pas foi ; son `bin/python` est un lien
symbolique vers un interpréteur du cache `uv`, souvent rompu. Lancer les tests
avec `.venv/bin/python` échoue donc, alors que `uv run` fonctionne.

Le lint suit le même principe (`uv run --with ruff`), configuré une seule fois
dans `ruff.toml` à la racine de la certification. Un réglage y est structurant
pour ce projet : `target-version = "py311"`. Sans lui, le formateur réécrit les
f-strings du notebook en syntaxe 3.12 et le fichier cesse de parser sous la
version qu'il déclare. Le cas s'est produit, `TestCoherenceDuNotebook` le
verrouille.

Le premier lancement télécharge les dépendances du notebook (PEP 723) dans le
sandbox Marimo. Les données ne sont pas versionnées : `data/raw/` est créé au
premier appel de `load_data()`, qui les récupère depuis
`https://cdn.freecodecamp.org/project-data/books/book-crossings.zip`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `recommender.py` | Logique : chargement, filtrage 200/100, matrice, `NearestNeighbors`, `get_recommends` |
| `notebook.py` | Notebook Marimo : exploration, visualisation, cellule de test officielle |
| `test_units.py` | Tests unitaires, dont la cellule de test officielle rejouée |
| `docs/enonce-freecodecamp.md` | Traduction de l'énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions que l'exercice fait travailler, expliquées |
| `data/raw/` | Jeu de données brut, non versionné : créé et rempli au premier lancement |

Deux fichiers vivent à la racine de la certification et servent aux cinq
projets : `ruff.toml` (lint et formatage) et `Makefile` (raccourcis).

## Le faux piège du projet

`kneighbors` renvoie les voisins par **distance croissante** : le plus proche en
premier. La cellule de test freeCodeCamp attend l'inverse, du plus éloigné au
plus proche. Le corrigé commence par « Catch 22 » (distance 0,794, la plus
lointaine) et finit par « The Vampire Lestat » (0,518, la plus proche). Sans
inversion, un modèle parfait échoue dès la première assertion
(`BookRecommender.recommend`, verrouillé par `test_units.py`).

## Le vrai piège : les notes à 0

Le jeu Book-Crossings enregistre aussi les interactions **sans note explicite**,
sous la forme d'une note à `0`, valeur pourtant absente de l'échelle 1-10
annoncée. Ce n'est pas un cas marginal : **62,3 % des notes brutes valent 0, et
74,6 % de celles retenues après filtrage**.

Conséquence sur le `fillna(0)` de `build_matrix` : une case à 0 ne dit plus
si l'utilisateur n'a pas noté le livre ou s'il lui a mis 0. Pour le calcul, cela
ne change rien : un 0 stocké et un 0 d'absence contribuent identiquement, c'est
à dire pas du tout, au produit scalaire comme à la norme. L'ambiguïté est donc
entièrement **interprétative**, et c'est ce qui la rend traître : aucun test ne
peut la détecter, seule la lecture des résultats en souffre.

Le décompte, maillon par maillon. Des 49 781 notes retenues, 264 disparaissent
au rattachement (ISBN absents du fichier des livres) et 381 doublons
utilisateur-titre sont écartés, laissant 49 136 notes. Parmi elles, 36 711
valent 0, d'où les 12 425 cellules non nulles de la matrice, sur 597 624 cases.
Autrement dit, les trois quarts de ce que le filtrage avait retenu comme
« notes » sont indiscernables du vide.

Ce qu'il faut en retenir pour lire les résultats : le modèle rapproche des
livres **touchés par les mêmes personnes**, pas des livres **appréciés
pareillement**. C'est de la co-interaction, pas de la similarité de goût. Le
choix est assumé parce que c'est celui du corrigé officiel, donc la condition
pour reproduire ses distances, pas parce qu'il modélise fidèlement le problème.

## Choix retenus

| Choix | Raison |
|---|---|
| `NearestNeighbors(metric="cosine", algorithm="brute")` | Imposé par l'énoncé. Cosinus = angle des profils, invariant par homothétie (pas par translation, voir `docs/`) ; `brute` est le seul algorithme acceptant cette métrique, les arbres KD/Ball exigeant une vraie distance métrique |
| Seuils ≥ 200 notes/utilisateur, ≥ 100 notes/livre | Imposés par l'énoncé ; mesuré : 673 livres × 888 utilisateurs, 49 781 notes retenues |
| Matrice dense float64, notes manquantes → 0 | Le 0 est neutre pour le produit scalaire ; reproduit le corrigé officiel à 4,4e-08 près (tolérance du test : 1e-4). Contrepartie assumée : le 0 devient ambigu, voir ci-dessous |
| Réunir les titres multi-ISBN et écarter les doublons (user, title) | Le jeu référence un même titre sous plusieurs ISBN : 50 titres concernés dans la matrice, 381 lignes écartées avant le pivot |
| Logique dans `recommender.py`, notebook orchestrateur | Convention reprise du projet 01 : le notebook n'héberge aucune logique |
| `get_recommends` fabriquée par `make_get_recommends(model)` | La fonction de contrat et le modèle restent découplés ; testable sans état global |

## État

Squelette opérationnel : données en place, pipeline validé contre le corrigé
officiel, 25 tests verts en moins de 15 s (voir « Lancement »), `ruff check` et
`ruff format --check` sans reproche. Les distances du corrigé sont reproduites
à 4,4e-08 près (écart dû à la version de scikit-learn et à l'ordre des
sommations flottantes, très sous la tolérance de 1e-4).

Reste à travailler dans le notebook, le diagnostic plutôt que la validation :

- le **biais de popularité** : déjà mesuré (corrélation de -0,227 entre le
  nombre de notes d'un livre et sa distance moyenne aux autres) et verrouillé
  par un test ; les livres très notés sont un peu plus proches de tout. Seule
  la figure reste à tracer ;
- la **robustesse aux seuils** : 200/100 sont imposés, mais faire varier le
  seuil dirait si « Catch 22 » reste la recommandation la plus lointaine ou si
  c'est un artefact du filtrage ;
- « Catch 22 » recommandé pour un roman de vampires est le résultat le plus
  intéressant du projet : probablement de la co-notation générique, ce que la
  section sur les zéros ci-dessus laisse attendre.

URL de soumission : *à compléter une fois le projet soumis.*
