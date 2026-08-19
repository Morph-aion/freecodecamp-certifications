# Notions mobilisées : Page View Time Series Visualizer

Ce que cet exercice fait travailler, notion par notion.

## Table des matières

- [1. Les séries temporelles](#1-les-séries-temporelles)
- [2. Le parsing de dates avec Pandas](#2-le-parsing-de-dates-avec-pandas)
- [3. Le filtrage des valeurs extrêmes](#3-le-filtrage-des-valeurs-extrêmes)
- [4. Les graphiques linéaires](#4-les-graphiques-linéaires)
- [5. Les graphiques en barres groupés](#5-les-graphiques-en-barres-groupés)
- [6. Les box plots et la décomposition](#6-les-box-plot-et-la-décomposition)

## 1. Les séries temporelles

Une **série temporelle** est une suite d'observations indexées par le temps. Ici,
ce sont les vues quotidiennes du forum freeCodeCamp, du 2016-05-09 au 2019-12-03
(1304 jours). Après filtrage des extrêmes, il reste 1238 jours, du 2016-05-19 au
2019-12-03.

**Éléments clés d'une série temporelle** :
- **Tendance (trend)** : mouvement à long terme (hausse, baisse, stable)
- **Saisonnalité (seasonality)** : motifs qui se répètent à intervalle régulier
  (hebdomadaire, mensuel, annuel)
- **Bruit (noise)** : variation aléatoire résiduelle

**Pourquoi c'est important ?** Les techniques d'analyse classiques (moyenne, variance)
supposent des données i.i.d. (indépendantes et identiquement distribuées). Les séries
temporelles violent cette hypothèse : une observation dépend des précédentes.

## 2. Le parsing de dates avec Pandas

```python
pd.read_csv('fcc-forum-pageviews.csv', parse_dates=['date'], index_col='date')
```

**`parse_dates=['date']`** : convertit automatiquement la colonne `date` en objet
`datetime64`. Sans cela, les dates restent des strings et les opérations temporelles
(mois, année, filtrage par plage) sont impossibles.

**`index_col='date'`** : utilise la colonne date comme index du DataFrame. C'est
essentiel pour les séries temporelles car Pandas optimise les opérations sur les
index DatetimeIndex (slicing par date, resample, rolling, etc.).

**Format de date** : Pandas détecte automatiquement le format (`YYYY-MM-DD`).
Si le format est ambigu (ex: `01/02/2020` = 1er février ou 2 janvier ?), il faut
spécifier `date_format` ou `dayfirst=True`.

## 3. Le filtrage des valeurs extrêmes

```python
df = df[(df['value'] >= df['value'].quantile(0.025)) & 
        (df['value'] <= df['value'].quantile(0.975))]
```

**Pourquoi filtrer ?** Les données de vues peuvent contenir des spikes (attaques
bot, événements viraux) qui masquent le signal réel. Filtrer les 2.5% extrêmes
de chaque côté conserve environ 95 % des données (94,9 % ici : 1238 lignes sur
1304) tout en éliminant les valeurs extrêmes.

**Quantiles vs percentiles** : `quantile(0.025)` = 2.5ᵉ percentile. C'est
exactement la même chose, le terme « quantile » est utilisé quand la fraction
est donnée en décimal (0 à 1) au lieu d'un pourcentage (0 à 100).

**Piège** : filtrer d'abord, puis calculer les stats. Sinon, les stats incluent
les valeurs qu'on veut supprimer.

## 4. Les graphiques linéaires

```python
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df.index, df['value'], color='crimson', linewidth=1)
```

**`fig, ax = plt.subplots()`** : crée une figure et un seul axes. C'est l'interface
orientée objet, plus flexible que `plt.plot()`.

**Paramètres de style** :
- `figsize=(15, 5)` : taille en pouces (largeur, hauteur)
- `color='crimson'` : couleur (nom CSS, hexadécimal, RGB)
- `linewidth=1` : épaisseur de la ligne

**Sérialisation des dates** : Matplotlib convertit automatiquement les `datetime`
en positions numériques. Les ticks de l'axe X sont adaptés à la plage de dates.

## 5. Les graphiques en barres groupés

```python
df_bar = df.groupby([df.index.year, df.index.month])['value'].mean()
df_bar = df_bar.unstack(level=1)
df_bar.plot(kind='bar', ax=ax)
```

**`df.index.year`** : accès direct aux composantes de la date via l'index
`DatetimeIndex`. Pas besoin de `apply(lambda x: x.year)`.

**`groupby([...]).mean()`** : calcule la moyenne des vues pour chaque combinaison
(année, mois).

**`unstack(level=1)`** : transforme l'index imbriqué en colonnes. Chaque mois
devient une colonne, ce qui permet de créer des barres groupées.

**`kind='bar'`** : type de graphique. Les barres sont automatiquement groupées
quand il y a plusieurs colonnes.

## 6. Les box plots et la décomposition

Un **box plot** (diagramme en boîte) affiche cinq statistiques clés :
- **Minimum** : moustache basse (ou point outlier)
- **Q1** (25ᵉ percentile) : bord inférieur de la boîte
- **Médiane** (Q2, 50ᵉ percentile) : ligne dans la boîte
- **Q3** (75ᵉ percentile) : bord supérieur de la boîte
- **Maximum** : moustache haute (ou point outlier)

**Pourquoi deux box plots ?**
- **Par année** : montre l'évolution de la distribution dans le temps (tendance)
- **Par mois** : montre la saisonnalité (les mois d'été ont-ils plus de vues ?)

**seaborn.boxplot** :
```python
df_box = df.copy()
df_box["year"] = df_box.index.year
df_box["month_name"] = df_box.index.month.map(lambda m: MONTHS_SHORT[m - 1])

sns.boxplot(x="year", y="value", data=df_box, ax=ax1)
sns.boxplot(x="month_name", y="value", data=df_box, ax=ax2, order=MONTHS_SHORT)
```

**Faut-il un `reset_index()` ?** Non, contrairement à ce qu'on lit souvent.
Seaborn résout `x=` et `y=` par nom, et accepte aussi bien une colonne ordinaire
que le nom de l'index : les deux formes ci-dessous produisent les mêmes quatre
boîtes.

```python
sns.boxplot(x="year", y="value", data=df_box)                 # DatetimeIndex conservé
sns.boxplot(x="year", y="value", data=df_box.reset_index())   # index numérique
```

Ce qui est réellement nécessaire, c'est que la grandeur portée en abscisse
**existe comme colonne** : `df.index.year` est un attribut de l'index, pas une
colonne, et ne peut donc pas être désigné par `x="year"` tant qu'il n'a pas été
matérialisé. C'est ce que fait la première ligne du bloc ci-dessus, et c'est
pourquoi ce projet n'appelle jamais `reset_index()`.

**`order=MONTHS_SHORT`** : sans cet argument, Seaborn ordonne les catégories par
ordre d'apparition dans les données, soit mai en premier (le jeu commence en mai
2016). Le correcteur attend janvier à décembre.

## Notions voisines, non implémentées ici

- **Rolling average** : `df.rolling(window=30).mean()` pour lisser une série.
- **Décomposition STL** : séparer tendance + saisonnalité + résidu.
- **Tests de stationnarité** : ADF test pour vérifier si la moyenne/variance sont constantes.
- **Forecasting** : ARIMA, Prophet, LSTM pour prédire les valeurs futures.
