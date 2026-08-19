# Notions mobilisées : Demographic Data Analyzer

Ce que cet exercice fait travailler, notion par notion.

## Table des matières

- [1. Pandas et les DataFrames](#1-pandas-et-les-dataframes)
- [2. Lecture de fichier CSV](#2-lecture-de-fichier-csv)
- [3. Comptage et value_counts](#3-comptage-et-value_counts)
- [4. Filtrage conditionnel](#4-filtrage-conditionnel)
- [5. Groupby et agrégation](#5-groupby-et-agrégation)
- [6. Calcul de pourcentages](#6-calcul-de-pourcentages)
- [7. String matching et idxmax](#7-string-matching-et-idxmax)

## 1. Pandas et les DataFrames

Pandas est la bibliothèque de référence pour l'analyse de données tabulaires en Python.
Un **DataFrame** est une structure de données en tableau (comme une feuille Excel) avec
des lignes et des colonnes nommées.

**DataFrame vs dict de listes** : un dict de listes (`{'col1': [1,2], 'col2': [3,4]}`)
fonctionne pour des données simples, mais manque d'opérations vectorisées, de méthodes
de groupby, de jointures, etc. Pandas fait tout cela et gère automatiquement l'index.

## 2. Lecture de fichier CSV

`pd.read_csv('adult.data.csv')` lit un fichier CSV et retourne un DataFrame.

**Le piège de ce projet** : le fichier `adult.data.csv` distribué par freeCodeCamp
**porte sa propre ligne d'en-tête**. Un `pd.read_csv(chemin)` sans option suffit.

La confusion est facile parce que la version d'origine du jeu de données, celle du
dépôt UCI, n'en a pas : c'est elle qu'on trouve décrite dans la plupart des tutoriels,
avec la liste des colonnes à passer en `names=[...]`. Appliquer cette recette au
fichier de freeCodeCamp transforme la ligne de titres en observation :

| | Sans option | Avec `header=None, names=[…]` |
|---|---|---|
| Lignes | 32 561 | 32 562 |
| `df["age"].dtype` | `int64` | `object` |
| Première valeur | `39` | `"age"` |

L'erreur ne se voit pas au chargement : elle éclate au premier calcul, sur
`TypeError: Cannot perform reduction 'mean' with string dtype`.

**Comment s'en prémunir** : plutôt que d'imposer les noms de colonnes, les vérifier
après lecture. C'est ce que fait `load_data()` : si l'en-tête ne correspond pas à ce
que le projet attend, elle lève une erreur qui nomme l'écart au lieu de laisser un
typage silencieusement faux se propager.

**Important** : Pandas détecte automatiquement les types (int, float, string) mais
peut se tromper sur les espaces en début/fin de chaîne. Utiliser `strip()` ou
`str.strip()` si nécessaire.

## 3. Comptage et value_counts

`df['race'].value_counts()` retourne une série avec le nombre d'occurrences de chaque
valeur unique, triée par ordre décroissant.

**C'est la réponse à la question 1** : combien de personnes de chaque race.

**Piège** : `value_counts()` ne compte pas les valeurs manquantes (`NaN`) par défaut.
Si des données manquantes sont présentes, il faut les gérer séparément.

## 4. Filtrage conditionnel

Les masks Pandas permettent de filtrer des lignes selon une condition :

```python
df[df['salary'] == '>50K']  # personnes gagnant plus de 50K
df[df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]  # éducation avancée
```

**Syntaxe** : `df[condition]` retourne un nouveau DataFrame ne contenant que les lignes
où la condition est `True`. La condition est un Series de booléens de même longueur
que le DataFrame.

**Opérateurs** : `&` (ET), `|` (OU), `~` (NON). Attention : parenthèses obligatoires
autour de chaque condition à cause de la priorité des opérateurs Python.

## 5. Groupby et agrégation

`df.groupby('colonne').agg({'autre_colonne': 'moyenne'})` regroupe les lignes par
valeur de la colonne et calcule une agrégation.

**C'est la réponse à la question 8** : le pays avec le plus haut pourcentage de
personnes gagnant >50K.

**Exemple** :
```python
(df[df['salary'] == '>50K']
 .groupby('native-country')
 .size() / df.groupby('native-country').size() * 100)
```

Calcule le pourcentage de personnes gagnant >50K par pays.

## 6. Calcul de pourcentages

Pas de fonction Pandas magique : `(part / total) * 100`.

**Piège** : division entière en Python 2. En Python 3, `/` est toujours une division
réelle, mais `//` reste une division entière. Pas de risque ici avec Python 3.

**Piège** : division par zéro. Si un groupe n'a aucune personne, le pourcentage est
indéfini. Utiliser `fillna(0)` ou filtrer les groupes vides.

## 7. `idxmax` : le label du maximum, pas le maximum

`Series.idxmax()` retourne l'**étiquette** de la valeur la plus grande, là où
`max()` retourne la valeur elle-même. Sur une série indexée par des noms de pays,
`idxmax()` donne donc le nom du pays.

**Le piège de la question 8** : elle demande le pays au plus haut *pourcentage* de
hauts revenus, pas celui qui en compte le plus. Les deux lectures ne donnent pas le
même pays, et l'écart est spectaculaire :

```python
# Compte brut : combien de personnes gagnent >50K dans chaque pays ?
df[df['salary'] == '>50K']['native-country'].value_counts().idxmax()
# -> 'United-States'  (7171 personnes, mais sur 29 170 résidents)

# Proportion : quelle part des habitants de chaque pays gagne >50K ?
rich = df[df['salary'] == '>50K'].groupby('native-country').size()
total = df.groupby('native-country').size()
(rich / total * 100).idxmax()
# -> 'Iran'  (41,9 %)
```

Les États-Unis dominent le compte brut simplement parce qu'ils représentent 90 % de
l'échantillon. C'est le biais d'effectif classique : une fréquence absolue n'est pas
une proportion. Le correcteur attend **Iran**, donc la seconde forme.

Noter la division `rich / total` : pandas aligne automatiquement les deux séries sur
leur index (les noms de pays), ce qui évite une jointure explicite.

**Pour la question 9**, en revanche, c'est bien un comptage brut qui est demandé :
« l'occupation la plus populaire » parmi les personnes gagnant >50K en Inde. Un
filtre sur les deux conditions, puis `value_counts().idxmax()` sur `occupation`.

## Notions voisines, non implémentées ici

- **Jointures (merge, join)** : combiner deux DataFrames sur une clé commune.
- **Pivot tables** : résumer des données en tableaux croisés dynamiques.
- **Apply / Applymap** : appliquer une fonction à des lignes, colonnes, ou cellules.
- **Gestion des données manquantes** : `isna()`, `fillna()`, `dropna()`.
