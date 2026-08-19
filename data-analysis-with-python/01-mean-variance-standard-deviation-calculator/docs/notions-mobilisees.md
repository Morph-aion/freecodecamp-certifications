# Notions mobilisées : Mean-Variance-Standard Deviation Calculator

Ce que cet exercice fait travailler, notion par notion.

## Table des matières

- [1. NumPy et les tableaux multi-dimensionnels](#1-numpy-et-les-tableaux-multi-dimensionnels)
- [2. Les axes de calcul](#2-les-axes-de-calcul)
- [3. La matrice aplatie (flatten)](#3-la-matrice-aplatie-flatten)
- [4. La reshape en matrice 3×3](#4-la-reshape-en-matrice-33)
- [5. Le vrai piège : quelle variance calcule NumPy ?](#5-le-vrai-piège--quelle-variance-calcule-numpy-)
- [6. Les erreurs de forme](#6-les-erreurs-de-forme)

## 1. NumPy et les tableaux multi-dimensionnels

NumPy est la bibliothèque fondamentale du calcul scientifique en Python. Elle fournit
un type `ndarray` (n-dimensional array) qui est beaucoup plus rapide et efficace en
mémoire que les listes Python pour les opérations numériques.

**Pourquoi NumPy et pas des listes Python ?** Une liste Python est un tableau de
pointeurs vers des objets Python arbitraires. Chaque opération doit vérifier le type
de chaque élément. Un `ndarray` NumPy stocke les données de manière contiguë en mémoire
avec un type uniforme, ce qui permet des opérations vectorisées (une seule boucle C
optimisée au lieu de boucles Python).

## 2. Les axes de calcul

Dans un tableau 2D (matrice), NumPy définit deux axes :
- **axe 0** : longe les lignes (de haut en bas) → calcule **par colonne**
- **axe 1** : longe les colonnes (de gauche à droite) → calcule **par ligne**

Exemple avec la matrice :
```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

- `np.mean(axis=0)` → `[4, 5, 6]` (moyenne de chaque colonne)
- `np.mean(axis=1)` → `[2, 5, 8]` (moyenne de chaque ligne)

**Piège courant** : inverser les axes. Si l'énoncé dit « mean along columns », il faut
utiliser `axis=0` (car on calcule en « écrasant » les lignes).

## 3. La matrice aplatie (flatten)

`np.mean(arr)` sans axe → calcule la moyenne de tous les éléments de la matrice
considérée comme un seul vecteur.

C'est équivalent à `np.mean(arr.flatten())` ou `np.mean(arr.ravel())`.

**flatten vs ravel** :
- `flatten()` retourne toujours une copie (plus sûr, plus lent)
- `ravel()` retourne une vue si possible (plus rapide, mais modifier la vue modifie l'original)

## 4. La reshape en matrice 3×3

`np.array(list).reshape(3, 3)` transforme une liste de 9 éléments en matrice 3 lignes × 3
colonnes.

**Contrainte** : le produit des dimensions doit être égal au nombre total d'éléments.
`reshape(3, 3)` fonctionne pour 9 éléments, mais échoue pour 8 ou 10.

**Ordre par défaut** : C-order (row-major), c'est-à-dire que les éléments remplissent les
lignes en premier : `[1,2,3,4,5,6,7,8,9]` → `[[1,2,3],[4,5,6],[7,8,9]]`.

## 5. Le vrai piège : quelle variance calcule NumPy ?

Il existe deux variances, et les deux bibliothèques les plus utilisées en science des
données n'ont pas le même défaut :

| | Diviseur | Nom | `ddof` |
|---|---|---|---|
| `numpy.var()` | `n` | variance de population | 0 (défaut) |
| `pandas.Series.var()` | `n − 1` | variance d'échantillon (correction de Bessel) | 1 (défaut) |

Sur la ligne `[0, 1, 2]`, l'écart est immédiat : NumPy donne **0,667**, pandas **1,0**.

Le correcteur freeCodeCamp attend les valeurs NumPy par défaut, donc `ddof=0`. Écrire
`arr.var()` est correct ; passer par pandas ou forcer `ddof=1` ferait échouer le test
sans que le calcul soit « faux » pour autant, seulement répondant à une autre question.

**Laquelle utiliser en dehors de cet exercice ?** `ddof=1` quand les données sont un
échantillon dont on veut estimer la variance de la population d'origine (le cas le plus
fréquent en statistique inférentielle), `ddof=0` quand les données *sont* la population
entière, ce qui est le cas ici : la matrice 3×3 est tout ce qui existe.

## 6. Les erreurs de forme

Quand la liste ne contient pas exactement 9 éléments, `reshape(3, 3)` lève un
`ValueError`. L'exercice demande de lever explicitement un `ValueError` avec un message
personnalisé avant même d'essayer la reshape.

**Pourquoi ?** Pour un message d'erreur clair et contrôlable par les tests unitaires.
La reshape est un opérateur « brut » qui lève des messages techniques peu lisibles.

## Notions voisines, non implémentées ici

- **axis=-1** : calcule sur le dernier axe (équivalent à axis=1 pour un tableau 2D).
  Pas nécessaire ici, mais courant dans d'autres contextes.
- **keepdims** : conserve la dimension réduite comme dimension de taille 1.
  Utile pour le broadcasting, pas nécessaire ici.
- **np.percentile / np.median** : autres mesures statistiques non demandées ici.
- **ndarray.T** : transposition de matrice (inverse lignes et colonnes).
