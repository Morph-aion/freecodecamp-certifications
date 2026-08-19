# 04 : Linear Regression Health Costs Calculator

Quatrième des 5 projets de la certification *Machine Learning with Python*.
Prédire des **coûts de santé** à l'aide d'un algorithme de **régression
linéaire**, sur le jeu de données fourni par freeCodeCamp. Seuil de réussite :
**MAE < 3500** (erreur absolue moyenne inférieure à 3500 dollars).

## Ce qui change par rapport au projet 03

Le projet 03 (KNN) traitait un problème **relationnel** : l'information utile
n'était pas dans chaque note isolée mais dans les chevauchements de goûts entre
profils. Ici, le problème est **tabulaire** : chaque observation est un vecteur
de features (âge, sexe, IMC, fumeur, nombre d'enfants, région), et la cible
est une variable continue (les dépenses de santé).

Le risque principal n'est plus la parcimonie (le jeu n'est pas creux) ni le
surapprentissage (la régression linéaire est un modèle simple), mais la
**qualité de la représentation** : les features catégorielles doivent être
encodées, et la relation entre features et cible peut être non linéaire
(notamment l'interaction fumeur × IMC).

## Lancement

```bash
uv run regression.py              # pipeline complet
uv run test_units.py              # tests unitaires
uv run --with marimo marimo edit --sandbox notebook.py            # notebook Marimo
```

`uv run test_units.py` plutôt que `python -m unittest` : seule cette forme lit
l'en-tête PEP 723 du fichier et résout ses dépendances.

Depuis la racine de la certification, le `Makefile` évite de retenir ces lignes :

```bash
make test PROJET=04-health-costs-regression   # la suite de ce projet
make lint                                     # ruff check + format --check
make fix                                      # corrige et formate
```

`make test` est paramétré par `PROJET` et `CERTIF` ; `make lint` porte sur
l'ensemble du dépôt.

Le script télécharge `insurance.csv` depuis le CDN freeCodeCamp au premier
lancement (`data/raw/` est gitignoré), entraîne le modèle, et vérifie que
MAE < 3500 sur le jeu de test. Les tests passent par `uv run` : le projet ne
versionne pas d'environnement Python, `uv` résout les dépendances à la volée.

## Fichiers

| Fichier | Rôle |
|---|---|
| `regression.py` | Pipeline complet : chargement, préparation, entraînement, évaluation |
| `test_units.py` | Tests unitaires : structure des données, encodage, interaction, seuil MAE |
| `notebook.py` | Notebook Marimo : exploration, visualisation, diagnostic |
| `docs/enonce-freecodecamp.md` | Traduction de l'énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions que l'exercice fait travailler, expliquées |
| `docs/quelle-regression-lineaire.md` | De quelle régression linéaire il s'agit exactement, et ce qu'elle n'est pas |
| `data/raw/` | Jeu de données brut, non versionné : créé et rempli au premier lancement |

## Le faux piège du projet

L'énoncé demande de « convertir les données catégorielles en nombres ». La
tentation est d'utiliser un encodage label (`LabelEncoder`) qui attribue un
entier à chaque catégorie (`female=0, male=1`). C'est **faux** : la régression
linéaire interprétera cet entier comme une valeur ordonnée, ce qui n'a pas de
sens pour un attribut binaire. Il faut un encodage one-hot (`pd.get_dummies`),
qui crée une colonne binaire par catégorie.

L'encodage one-hot crée une colonne redondante (par exemple `sex_female` et
`sex_male` sont colinéaires). La régression linéaire avec intercept peut
provoquer une matrice singulière. La solution est de supprimer une des colonnes
(`drop_first=True`) ou de ne pas inclure l'intercept.

## Le vrai piège : l'interaction fumeur × IMC

La relation entre IMC et dépenses n'est pas linéaire : elle est **massivement
modulée par le statut de fumeur**. Un fumeur avec un IMC élevé a des dépenses
beaucoup plus élevées qu'un non-fumeur avec le même IMC. Sans feature
d'interaction, le modèle sous-estime systématiquement les fumeurs à IMC élevé
et surestime les non-fumeurs.

L'ajout d'une colonne `bmi_smoker = bmi × smoker_yes` réduit significativement
la MAE et améliore l'ajustement du modèle.

## Le nom de la colonne cible : `expenses` ou `charges`

Le jeu de données officiel distribué par freeCodeCamp nomme la cible
**`expenses`**, conformément à l'énoncé. La version largement diffusée sur
Kaggle nomme la même colonne **`charges`**, et arrondit `bmi` à deux décimales
au lieu d'une. Un code qui code `charges` en dur casse sur le jeu officiel, et
inversement.

`resolve_target()` détecte le nom présent et le reste du pipeline s'y adapte.
Les tests acceptent les deux, ce qui rend le projet exécutable indifféremment
sur l'un ou l'autre jeu.

Autre particularité du fichier officiel : ses fins de ligne sont des **CR
seuls** (convention Mac classique, antérieure à Mac OS X). `pandas.read_csv`
les gère sans configuration, mais les outils en ligne (`wc -l`) comptent 0 ligne.

## Choix retenus

| Choix | Raison |
|---|---|
| Régression linéaire (`LinearRegression`) | Imposé par l'énoncé |
| Encodage one-hot (`pd.get_dummies`) | Les catégorielles n'ont pas d'ordre ; l'encodage label fausse l'interprétation |
| `drop_first=True` | Évite la colinéarité avec l'intercept |
| Feature d'interaction `bmi × smoker` | Capture l'effet combiné IMC-fumeur, principal levier de performance |
| 80/20 split, `random_state=42` | Imposé par l'énoncé ; reproductibilité mesurée |

## Approches publiques comparées

Ce projet a été résolu sans consulter de solution existante. Une recherche
menée après coup situe ce qui a été écrit ici, et attribue ce qui vient
d'ailleurs.

**L'interaction `bmi × smoker` est le geste canonique de ce jeu de données, pas
une trouvaille.** `insurance.csv` est le dataset du livre *Machine Learning with
R* de Brett Lantz, dont le chapitre sur la régression introduit précisément
cette interaction comme démonstration pédagogique. Le dataset compte plus de
2000 notebooks publics sur [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance),
où l'interaction est le premier réflexe documenté.

L'amplitude du gain se retrouve à l'identique ailleurs : un article
[Towards Data Science](https://towardsdatascience.com/medical-cost-prediction-4876e3449adf/)
rapporte une MAE de 3941 sans interaction contre 2835 avec, à comparer aux
4190 → 2757 mesurés ici. Ce n'est donc pas un artefact de ce code : c'est une
propriété du jeu de données.

Une différence subsiste : la variante de Lantz est **binaire**
(`bmi30 = 1 si IMC > 30`, puis `bmi30 × smoker`), celle retenue ici est
**continue** (`bmi × smoker_yes`). La version continue évite un seuil arbitraire
et laisse l'effet croître régulièrement avec l'IMC ; c'est la moins répandue des
deux.

**Ce qui n'a pas d'équivalent trouvé** : l'usage de scikit-learn plutôt que
Keras (les solutions freeCodeCamp empilent des couches denses, l'énoncé
suggérant `model.evaluate`), la vérification sur 200 graines que le seuil est
raté sans l'interaction, et la documentation du coefficient négatif de
`smoker_yes`. Aucune discussion du forum freeCodeCamp consultée ne mentionne
l'interaction : la communauté atteint le seuil en ajoutant des couches, pas en
travaillant les variables.

## État : objectif atteint, MAE 2757 $

Pipeline validé sur le jeu de données officiel freeCodeCamp, seuil de 3500 $
tenu avec une marge de 743 $.

| Métrique | Valeur |
|---|---|
| MAE | **2757 $** (seuil : < 3500) |
| RMSE | 4574 $ |
| R² | 0,865 |

### L'interaction n'est pas un artefact du `random_state`

Le seuil est atteint avec `random_state=42`, mais un seul split ne prouve rien.
Mesure sur 50 graines différentes :

| Modèle | MAE moyenne | MAE max | Échecs du seuil |
|---|---|---|---|
| sans `bmi_smoker` | 4190 $ | 4704 $ | **50/50** |
| avec `bmi_smoker` | 2903 $ | 3389 $ | 0/50 |

L'interaction améliore la MAE sur **50 graines sur 50**. Sans elle, le projet
échouerait systématiquement au seuil, quel que soit le split retenu : ce n'est
pas un gain marginal mais la condition de réussite de l'exercice.

Le livrable freeCodeCamp est un notebook Google Colab : `notebook.py` (Marimo)
tient le raisonnement et les figures, il reste à en porter le contenu dans le
notebook Colab officiel pour la soumission.

URL de soumission : *à compléter une fois le projet soumis.*