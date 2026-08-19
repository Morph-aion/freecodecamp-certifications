# Notions mobilisées par le projet

Ce que cet exercice fait travailler, notion par notion, avec le lien vers le code qui
l'implémente. L'énoncé freeCodeCamp donne les instructions sans jamais les expliquer : ce document
comble cet écart.

Distinct de [enonce-freecodecamp.md](enonce-freecodecamp.md), qui est le cahier des charges
imposé et ne doit pas être commenté.

Les références `regression.py:NN` pointent sur les lignes exactes au moment de la rédaction. Elles
se décalent dès que le fichier est modifié : en cas de doute, chercher le nom de la fonction ou de
la constante plutôt que de se fier au numéro.

## Table des matières

- [1. La régression linéaire multiple](#1-la-régression-linéaire-multiple)
- [2. L'encodage one-hot et le piège du label encoding](#2-lencodage-one-hot-et-le-piège-du-label-encoding)
- [3. La colinéarité et `drop_first`](#3-la-colinéarité-et-drop_first)
- [4. L'interaction fumeur × IMC](#4-linteraction-fumeur--imc)
- [5. Le split entraînement / test](#5-le-split-entraînement--test)
- [6. Les métriques d'évaluation](#6-les-métriques-dévaluation)
- [7. Notions voisines, non implémentées ici](#7-notions-voisines-non-implémentées-ici)

## 1. La régression linéaire multiple

`LinearRegression()` de scikit-learn : [regression.py:114](../regression.py#L114).

**Le principe** : on cherche une fonction linéaire `y = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ` qui
minimise la somme des erreurs au carré (moindres carrés ordinaires). Chaque coefficient `βᵢ`
mesure l'effet marginal de la feature `xᵢ` sur la cible `y`, toutes les autres features étant
égales.

**Ce que fait `fit`** : résout le système normale `XᵀXβ = Xᵀy`, qui admet une solution unique
dès que `XᵀX` est inversible (pas de multicolinéarité parfaite). Pas d'hyperparamètre, pas
d'itération : c'est une résolution analytique directe.

**Limite** : la relation entre features et cible est supposée linéaire. Si la relation est
réellement non linéaire (comme l'interaction IMC × fumeur), il faut la modéliser explicitement
en créant une nouvelle feature.

## 2. L'encodage one-hot et le piège du label encoding

`pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)` :
[regression.py:75](../regression.py#L75).

**Le piège** : l'énoncé dit « convertir les données catégorielles en nombres ». La solution
la plus simple est `LabelEncoder`, qui transforme `female→0, male→1`. Mais la régression
linéaire interprète ces entiers comme des **valeurs ordonnées** : elle supposera que
`male > female` et que la différence entre les deux est de 1. C'est absurde pour un attribut
sans hiérarchie.

**La solution** : l'encodage one-hot crée une colonne binaire par catégorie. `sex_male` vaut 1
pour les hommes, 0 pour les femmes. Pas d'ordre implicite, pas de distance artificielle.

**Pourquoi `drop_first=True`** : sans suppression, les colonnes `sex_female` et `sex_male`
sont parfaitement colinéaires (leur somme vaut toujours 1). Cette redondance rend la matrice
`XᵀX` singulière, ce qui provoque une erreur ou des coefficients instables. Supprimer une
colonne par groupe résout le problème sans perdre d'information.

## 3. La colinéarité et `drop_first`

Voir point 2 ci-dessus. Formalisation : si `k` catégories → `k-1` colonnes après suppression.
L'intercept `β₀` joue le rôle de la catégorie de référence (celle qui a été supprimée).

Dans ce projet : `sex` (2 catégories → 1 colonne), `smoker` (2 → 1), `region` (4 → 3).
Total : 5 colonnes catégorielles au lieu de 8.

## 4. L'interaction fumeur × IMC

`df["bmi_smoker"] = df["bmi"] * df["smoker_yes"]` : [regression.py:80](../regression.py#L80).

**Le constat** : la relation entre IMC et dépenses n'est pas la même pour les fumeurs et
les non-fumeurs. Sur un nuage de points IMC vs charges coloré par le statut de fumeur, on
voit deux nuages distincts : les non-fumeurs avec une pente douce, les fumeurs avec une pente
beaucoup plus raide.

**Ce que fait l'interaction** : la colonne `bmi_smoker` vaut `bmi` pour les fumeurs et `0`
pour les non-fumeurs. Le modèle apprend alors `charges = β₀ + β₁·bmi + β₂·smoker_yes + β₃·bmi·smoker_yes + …`
Pour un fumeur : `charges = β₀ + β₁·bmi + β₂ + β₃·bmi = (β₀ + β₂) + (β₁ + β₃)·bmi`.
La pente effective de l'IMC est `β₁ + β₃` pour un fumeur, `β₁` pour un non-fumeur.

**Impact mesuré** : sans interaction, MAE ≈ 4200 $. Avec interaction, MAE ≈ 2800 $. L'ajout
de cette seule colonne réduit l'erreur de plus de 30 %.

Vérifié sur 50 graines de split différentes : le modèle sans interaction dépasse le seuil de
3500 $ dans **50 cas sur 50** (MAE minimale 3695 $), celui avec interaction reste sous le seuil
dans **50 cas sur 50** (MAE maximale 3389 $). L'interaction n'est donc pas un gain obtenu par
chance sur `random_state=42` : c'est ce qui rend l'exercice réalisable.

**Le prix à payer : les coefficients ne se lisent plus isolément.** C'est la contrepartie de
l'interaction, et elle surprend à la première lecture des résultats. Coefficients obtenus :

| Terme | Coefficient |
|---|---|
| `smoker_yes` | **−21 213** |
| `bmi_smoker` | +1 471 |

Le coefficient du tabac est négatif. Lu seul, il suggérerait que fumer fait *baisser* les
dépenses, ce qui est absurde. L'erreur consiste à interpréter `β₂` comme « l'effet fumeur »
alors qu'il n'en est que l'ordonnée à l'origine : l'effet réel est `β₂ + β₃·bmi`, soit
+8 205 $ à un IMC de 20, +22 913 $ à 30, +37 622 $ à 40. Positif partout, et croissant.

Le point où l'effet s'annulerait (`bmi = −β₂/β₃ ≈ 14,4`) se situe **hors de la plage observée**
(IMC minimum du jeu : 16,0). Le modèle n'y prédit rien : c'est une extrapolation, pas un
résultat.

Règle générale : dès qu'un terme d'interaction est présent, le coefficient d'une variable
principale ne vaut que pour le cas où l'autre variable est nulle. Les modèles additifs simples
n'ont pas ce problème, ce qui explique qu'on l'oublie facilement en passant aux seconds.

## 5. Le split entraînement / test

`train_test_split(test_size=0.2, random_state=42)` : [regression.py:100](../regression.py#L100).

**Pourquoi 80/20** : imposé par l'énoncé. L'idée générale est de garder une partie des données
jamais vues par le modèle pour mesurer sa capacité de généralisation.

**Pourquoi `random_state=42`** : reproductibilité. Sans graine fixe, le split change à chaque
exécution et le score fluctue, rendant la comparaison de choix arbitraire.

**Risque** : avec seulement 1338 lignes, le jeu de test contient ~268 observations. La MAE
mesurée est donc elle-même incertaine. Mesuré par bootstrap (4000 rééchantillonnages du jeu de
test) : MAE 2757 $, intervalle de confiance à 95 % **[2341, 3234]**, soit une demi-largeur de
**±446 $**. L'incertitude porte sur près d'un cinquième de la valeur mesurée.

Ce n'est pas un problème pour la certification (la borne haute de l'intervalle reste sous le
seuil de 3500), mais cela veut dire qu'un écart de quelques dizaines de dollars entre deux
variantes du modèle n'est pas interprétable : il est noyé dans le bruit d'échantillonnage.

Le choix de `random_state=42` illustre le même point. Sur 200 graines, la MAE varie de 2546 $ à
3389 $ (médiane 2923 $) ; la graine 42 tombe au 15ᵉ percentile, donc parmi les splits
légèrement favorables. Cela ne fausse aucune conclusion (le seuil est tenu sur **200 graines
sur 200**), mais c'est la raison pour laquelle les comparaisons de ce document s'appuient sur
des distributions plutôt que sur une exécution unique.

## 6. Les métriques d'évaluation

`evaluate_model()` : [regression.py:121](../regression.py#L121).

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** (*mean absolute error*) | mean(\|y - ŷ\|) | Erreur absolue moyenne, en dollars. **C'est le critère freeCodeCamp** |
| RMSE (*root mean squared error*) | √(mean((y - ŷ)²)) | Racine de l'erreur quadratique moyenne : pénalise davantage les grosses erreurs |
| R² (*coefficient of determination*) | 1 - SS_res / SS_tot | Coefficient de détermination : proportion de variance expliquée (1 = parfait) |

Les sigles restent sous leur forme anglaise, c'est ainsi qu'ils apparaissent dans
scikit-learn (`mean_absolute_error`, `r2_score`) et dans la littérature ; mais le nom
développé est donné une fois pour lever l'ambiguïté.

**Pourquoi la MAE et pas le R²** : le seuil de 3500 $ est un critère métier (« prédire les
coûts à moins de 3500 $ près »). Le R² mesure la qualité statistique du modèle mais ne se
traduit pas directement en dollars.

## 7. Notions voisines, non implémentées ici

- **Ridge / Lasso** : régression régularisée, utile quand le nombre de features est grand ou
  qu'on soupçonne de la multicolinéarité. Ici, 9 features pour 1338 lignes : pas besoin.
- **PolynomialFeatures** : générer automatiquement des termes d'interaction et de degré 2.
  Fonctionnerait, mais le choix manuel de `bmi × smoker` est plus interprétable et suffisant.
- **StandardScaler** : la régression linéaire ne nécessite pas la normalisation des features
  (contrairement à k-NN ou SGD). Les coefficients changent d'échelle, mais les prédictions
  et la MAE restent identiques.
