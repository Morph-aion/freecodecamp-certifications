# Notions mobilisées : Medical Data Visualizer

Ce que cet exercice fait travailler, notion par notion.

## Table des matières

- [1. Matplotlib et les figures](#1-matplotlib-et-les-figures)
- [2. Les graphiques catégoriels avec seaborn](#2-les-graphiques-catégoriels-avec-seaborn)
- [3. Les heatmaps de corrélation](#3-les-heatmaps-de-corrélation)
- [4. Le nettoyage des données](#4-le-nettoyage-des-données)
- [5. La création de features dérivées](#5-la-création-de-features-dérivées)
- [6. La normalisation des variables binaires](#6-la-normalisation-des-variables-binaires)

## 1. Matplotlib et les figures

Matplotlib est la bibliothèque de base de la visualisation en Python. Tout autre
outil (Seaborn, Plotly, etc.) construit sur ses concepts fondamentaux.

**Hiérarchie** :
- `Figure` : le canevas complet (fenêtre ou image)
- `Axes` : un graphique individuel dans la figure (une figure peut en contenir plusieurs)
- `Axis` : les axes x et y d'un graphique

**Deux interfaces** :
- `plt.plot()` : interface imperative (simple pour des graphiques rapides)
- `fig, ax = plt.subplots()` : interface orientée objet (plus de contrôle)

Le projet utilise `sns.catplot()` qui retourne un `FacetGrid`, ce qui implique
une gestion légèrement différente de la figure.

## 2. Les graphiques catégoriels avec seaborn

`sns.catplot(x=..., y=..., hue=..., col=..., kind='count', data=...)` crée un
graphique de comptage par catégorie.

**Paramètres clés** :
- `x` : variable catégorielle pour l'axe X
- `y` : variable numérique pour l'axe Y (optionnel pour `kind='count'`)
- `hue` : variable de coloration (sépare les barres)
- `col` : variable de colonne (sépare les panneaux)
- `kind` : type de graphique (`'count'`, `'bar'`, `'box'`, etc.)

**Pourquoi catplot et pas plt.bar ?** `catplot` gère automatiquement le comptage,
le groupby, et la création de sous-graphiques. C'est plus haut niveau et plus
adapté aux données catégorielles.

## 3. Les heatmaps de corrélation

`sns.heatmap(corr, annot=True, fmt='.1f', linewidths=0.5)` affiche une matrice
de corrélation sous forme de carte thermique.

**La matrice de corrélation** : matrice carrée où chaque cellule contient le
coefficient de corrélation de Pearson entre deux variables. Valeurs entre -1
(corrélation négative parfaite) et +1 (corrélation positive parfaite), 0 = pas
de corrélation linéaire.

**Masquer le triangle supérieur** : la matrice de corrélation est symétrique
(`corr(A,B) = corr(B,A)`). Masquer la moitié supérieure évite la redondance.

```python
import numpy as np
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f')
```

**`np.triu`** : retourne le triangle supérieur d'une matrice. `np.ones_like` crée
une matrice de 1 de la même forme. Le masque est `True` pour les cellules à masquer.

## 4. Le nettoyage des données

Avant toute visualisation, il faut filtrer les données aberrantes :

- **Pression artérielle** : `ap_lo ≤ ap_hi` (la diastolique ne peut pas être
  supérieure à la systolique)
- **Taille** : entre le 2.5ᵉ et 97.5ᵉ percentile (supprimer les extrêmes)
- **Poids** : entre le 2.5ᵉ et 97.5ᵉ percentile

**Pourquoi ces filtres ?** Les données médicales contiennent souvent des erreurs
de saisie. Un patient avec `ap_lo=200` et `ap_hi=10` a manifestement une erreur.
Ces filtres suppriment les lignes manifestement fausses sans toucher aux données
réellement extrêmes.

**Percentiles vs valeurs absolues** : les percentiles s'adaptent à la distribution
des données. Si la plupart des gens mesurent entre 150cm et 200cm, le 2.5ᵉ percentile
sera autour de 150cm. C'est plus robuste que de fixer des seuils arbitraires.

## 5. La création de features dérivées

Créer la colonne `overweight` à partir de `height` et `weight` :

```python
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)
```

**IMC (Indice de Masse Corporelle)** : `weight / height²` où height est en mètres.
Un IMC > 25 est classé comme « surpoids » par l'OMS.

**`.astype(int)`** : convertit le booléen en entier (0 ou 1). Les graphiques
catégoriels de Seaborn attendent des valeurs numériques ou des strings, pas des booléens.

## 6. La normalisation des variables binaires

```python
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
```

**Pourquoi ?** Les colonnes `cholesterol` et `gluc` contiennent des valeurs 1, 2, 3
qui représentent des niveaux. L'énoncé demande de les binariser : 0 = normal (1),
1 = anormal (>1).

**Piège** : ne pas confondre « binariser » (0/1) avec « normaliser » (mettre à
l'échelle [0,1]). Ici c'est une binarisation par seuil.

## Notions voisines, non implémentées ici

- **Pair plots** : `sns.pairplot(df)` affiche toutes les relations deux à deux.
- **Violin plots** : `sns.violinplot()` combine box plot et densité de kernel.
- **Joint plots** : `sns.jointplot()` affiche la relation entre deux variables
  avec distributions marginales.
- **Subplots manuels** : `fig, axes = plt.subplots(nrows, ncols)` pour un contrôle
  total de la disposition.
