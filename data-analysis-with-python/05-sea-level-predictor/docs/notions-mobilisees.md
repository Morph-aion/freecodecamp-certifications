# Notions mobilisées : Sea Level Predictor

Ce que cet exercice fait travailler, notion par notion.

## Table des matières

- [1. La régression linéaire simple](#1-la-régression-linéaire-simple)
- [2. scipy.stats.linregress](#2-scipystatslinregress)
- [3. L'extrapolation](#3-lextrapolation)
- [4. Le scatter plot](#4-le-scatter-plot)
- [5. Deux modèles, deux prédictions](#5-deux-modèles-deux-prédictions)

## 1. La régression linéaire simple

La régression linéaire simple modélise la relation entre deux variables continues :
`y = β₀ + β₁·x + ε`

- **y** : variable dépendante (CSIRO Adjusted Sea Level)
- **x** : variable indépendante (Year)
- **β₀** : ordonnée à l'origine (intercept), valeur de y quand x = 0
- **β₁** : pente, variation de y pour une unité de x
- **ε** : terme d'erreur (résidu)

**Hypothèses** :
- Linéarité : la relation entre x et y est linéaire
- Indépendance des résidus
- Homoscédasticité : variance constante des résidus
- Normalité des résidus (pour les intervalles de confiance)

**Ici** : on suppose que le niveau de la mer augmente linéairement avec le temps.
C'est une approximation raisonnable sur une courte période, mais le réchauffement
accélère, donc la relation est probablement non linéaire sur le long terme.

## 2. scipy.stats.linregress

```python
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(x, y)
```

**Valeurs retournées** :
- `slope` : pente (β₁) : variation du niveau de la mer par an (en pouces/an)
- `intercept` : ordonnée à l'origine (β₀)
- `r_value` : coefficient de corrélation de Pearson (r) : force de la relation linéaire
- `p_value` : p-value du test de significativité (H₀: β₁ = 0)
- `std_err` : erreur standard de la pente

**r² (coefficient de détermination)** : `r_value ** 2`, proportion de la variance
de y expliquée par x. r² = 0.98 signifie que 98% de la variation du niveau de la mer
est expliquée par le temps.

**Pourquoi linregress et pas LinearRegression ?** `linregress` est une fonction
simple pour la régression bivariée. `sklearn.linear_model.LinearRegression` est
plus général (multivarié, plus d'options), mais surdimensionné ici.

## 3. L'extrapolation

L'extrapolation consiste à prédire en dehors de la plage des données observées.
Ici, on prédit le niveau de la mer en 2050 alors que les données s'arrêtent en 2013.

**Risque** : l'extrapolation suppose que le modèle reste valide en dehors de la
plage observée. Si le taux d'élévation accélère (comme c'est le cas avec le
réchauffement climatique), la prédiction linéaire sous-estimera la réalité.

**C'est acceptable ici** : l'exercice demande explicitement une prédiction linéaire.
Le but est de montrer la capacité à utiliser `linregress` et à étendre une ligne
de tendance, pas de faire de la climatologie sérieuse.

## 4. Le scatter plot

```python
ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], alpha=0.5, label='Raw Data')
```

**`alpha=0.5`** : transparence. Quand il y a beaucoup de points superposés,
la transparence permet de voir les zones de forte densité.

**Pourquoi scatter et pas line ?** Les données sont des mesures individuelles
avec du bruit. Un scatter plot montre la distribution réelle, un line plot
suggérerait une continuité qui n'existe pas nécessairement.

## 5. Deux modèles, deux prédictions

Le projet demande deux lignes de tendance :
1. **Toutes les données (1880-2013)** : tendance à long terme
2. **Données récentes (2000-2013)** : tendance actuelle

**Pourquoi deux modèles ?** Si le taux d'élévation a changé (accéléré, ralenti),
un seul modèle sur toutes les données masquerait ce changement. Comparer les deux
pentes permet de détecter une accélération.

**Exemple** :
- Pente globale : +0.06 pouces/an
- Pente depuis 2000 : +0.12 pouces/an
- Conclusion : le taux a doublé, ce qui est cohérent avec les données climatiques.

## Notions voisines, non implémentées ici

- **Régression polynomiale** : `np.polyfit(x, y, 2)` pour des relations non linéaires.
- **Intervalles de confiance** : calculer les bornes de prédiction autour de la ligne.
- **Métriques d'évaluation** : R², RMSE, MAE pour évaluer la qualité du modèle.
- **Régression multivariée** : ajouter d'autres variables (température, CO₂) pour
  améliorer la prédiction.
