# Quelle régression linéaire, exactement ?

L'énoncé dit « un algorithme de régression », le README dit « régression linéaire ». Ces deux
formulations recouvrent une famille entière de modèles, et le projet en implémente un seul,
précis. Ce document nomme lequel, et situe les voisins qu'il n'implémente pas.

## Réponse courte

Le modèle du projet est une **régression linéaire multiple, estimée par moindres carrés
ordinaires, avec variables indicatrices et un terme d'interaction**.

En notation statistique classique :

```
charges = β₀ + β₁·age + β₂·bmi + β₃·children
        + β₄·sex_male + β₅·smoker_yes
        + β₆·region_northwest + β₇·region_southeast + β₈·region_southwest
        + β₉·(bmi × smoker_yes) + ε
```

Neuf features, un intercept. `LinearRegression()` de scikit-learn résout ce système par
moindres carrés ordinaires (*ordinary least squares*, OLS).

## Pourquoi « multiple » et pas « simple »

La **régression linéaire simple** n'a qu'une seule variable explicative : `y = β₀ + β₁x`.
C'est le cas d'école qui se trace sur un plan.

Ici il y a neuf variables explicatives, donc **multiple**. La distinction n'est pas cosmétique :
en régression multiple, chaque coefficient `βᵢ` s'interprète *toutes autres variables tenues
constantes*, ce qui n'a pas de sens en régression simple. C'est aussi ce qui rend possible la
multicolinéarité, et donc nécessaire le `drop_first=True` de l'encodage one-hot.

## « Linéaire » porte sur les coefficients, pas sur les variables

C'est le point qui prête le plus à confusion, et le terme d'interaction du projet l'illustre
directement.

Le modèle reste **linéaire en les paramètres** : il est une somme de termes de la forme
`βᵢ × (quelque chose)`. Ce « quelque chose » peut être n'importe quelle transformation des
données brutes, y compris un produit de deux variables.

`bmi_smoker = bmi × smoker_yes` est un produit de deux variables, donc une relation **non
linéaire en les données**. Le modèle l'absorbe pourtant sans cesser d'être une régression
linéaire, parce que le coefficient `β₉` multiplie ce produit de façon linéaire. C'est un modèle
linéaire ajusté sur des features non linéaires, pas un modèle non linéaire.

Conséquence pratique : la pente de l'IMC n'est plus constante. Elle vaut `β₂` pour un
non-fumeur, `β₂ + β₉` pour un fumeur (détail dans [notions-mobilisees.md](notions-mobilisees.md)
§4). Le modèle capture une non-linéarité tout en restant dans la famille linéaire.

## Ce que le projet n'est pas

| Famille | Ce qui la distingue | Pourquoi pas ici |
|---|---|---|
| Régression linéaire **simple** | Une seule variable explicative | 9 features dans le projet |
| Régression **polynomiale** | Termes `x²`, `x³`… ajoutés explicitement | Reste une régression linéaire, non testée ici ; `PolynomialFeatures` la générerait |
| **Ridge / Lasso / ElasticNet** | Moindres carrés **pénalisés** (L2 / L1) | 9 features pour 1338 lignes : pas de problème de dimension ni de colinéarité résiduelle |
| **GAM** (modèle additif généralisé) | Chaque variable passe par une fonction lisse `f(x)` estimée (splines) | Sort de la famille linéaire au sens strict ; mesuré ci-dessous |
| **GLM** (modèle linéaire généralisé) | Cible non gaussienne via une fonction de lien (Gamma, Poisson…) | Recommandé en théorie pour des coûts ; mesuré plus bas, et moins bon ici |
| Régression **robuste** (Huber, RANSAC) | Pondère les points aberrants | Les résidus restent asymétriques, mais la MAE tient le seuil avec de la marge |
| **Arbres / forêts / gradient boosting** | Pas de forme fonctionnelle imposée | L'énoncé demande une régression |

## Ce qu'un GAM apporterait, mesuré

Le GAM est la famille la plus proche : il garde l'additivité (`y = f₁(x₁) + f₂(x₂) + …`) mais
remplace chaque coefficient par une fonction lisse. Mesure sur le pipeline du projet, en
validation croisée 5-fold :

| Modèle | MAE (5-fold) |
|---|---|
| linéaire multiple, sans interaction | 4203 $ |
| **linéaire multiple + interaction (le projet)** | **2921 $** |
| splines cubiques sur `age` et `bmi` (approche GAM) | 2861 $ |

Lecture : le passage aux splines gagne 60 $, soit environ 2 %. À comparer aux 1282 $ gagnés par
le seul terme d'interaction, et surtout à l'incertitude de mesure (±446 $ en bootstrap sur le
holdout). **Le gain du GAM est noyé dans le bruit ; celui de l'interaction ne l'est pas.**

C'est le résultat intéressant du projet : la non-linéarité qui compte ici n'est pas la courbure
de chaque variable prise isolément (ce que le GAM modélise), mais l'**interaction** entre deux
variables (ce que le GAM additif standard ne modélise pas non plus). Ajouter de la souplesse au
mauvais endroit ne sert à rien.

## Le principe hiérarchique

Le modèle contient `bmi`, `smoker_yes` **et** leur produit. Ce n'est pas un détail : c'est le
*principe hiérarchique*, qui veut qu'un modèle incluant une interaction inclue aussi les effets
principaux correspondants, même si leurs coefficients sont individuellement non significatifs
(ISLR §3.3.2).

La raison est directement visible ici. Toute la lecture « la pente de l'IMC vaut `β₂` pour un
non-fumeur, `β₂ + β₉` pour un fumeur » suppose que `β₂` existe dans le modèle. Retirer l'effet
principal `bmi` forcerait la pente des non-fumeurs à zéro et changerait la signification même du
terme d'interaction, qui est corrélé aux deux variables dont il est le produit.

## Les hypothèses du modèle, et lesquelles comptent vraiment

Le théorème de **Gauss-Markov** garantit que l'estimateur des moindres carrés est BLUE (*best
linear unbiased estimator*) sous cinq hypothèses : linéarité en les paramètres, échantillonnage
aléatoire, absence de colinéarité parfaite, exogénéité (`E[u|x] = 0`) et homoscédasticité.

Trois précisions que la formulation courante escamote :

**« Best » veut dire : de variance minimale parmi les estimateurs linéaires sans biais.** Pas
« le meilleur estimateur possible ». Un estimateur biaisé comme Ridge peut avoir une erreur
quadratique plus faible. C'est exactement l'argument du compromis biais-variance qui justifie la
pénalisation.

**La normalité des résidus ne fait pas partie de ces hypothèses.** Elle n'intervient nulle part
dans la démonstration, qui n'utilise que des moments d'ordre 1 et 2. Elle sert à deux choses
seulement : les distributions exactes des tests t et F en petit échantillon (avec n = 1338, le
théorème central limite rend la question sans objet), et les **intervalles de prédiction**,
seul endroit où la non-normalité mord quel que soit n, puisqu'un intervalle gaussien est
symétrique alors que la distribution des coûts ne l'est pas.

**Les violations n'ont pas toutes les mêmes conséquences.** C'est la distinction utile :

| Hypothèse violée | Biaise les coefficients | Invalide tests et erreurs standard | Dégrade la prédiction |
|---|---|---|---|
| Linéarité (mauvaise spécification) | oui | oui | **oui** |
| Exogénéité | oui | oui | oui |
| Homoscédasticité | non | **oui** | non |
| Normalité | non | en petit échantillon seulement | non |
| Colinéarité imparfaite | non | oui (variances gonflées) | non |

Le projet ne produit ni p-values ni intervalles de confiance sur les coefficients : il mesure une
MAE hors échantillon. La seule ligne qui menacerait ce critère est la première, la mauvaise
spécification, précisément ce que le terme d'interaction corrige.

## Ce que les tests disent réellement sur ce jeu de données

Le sens commun veut qu'une variable de coût soit hétéroscédastique : la dispersion croît avec le
niveau. Mesuré, c'est plus intéressant que ça.

| Test | Sans interaction | Avec interaction |
|---|---|---|
| Breusch-Pagan (p) | 1,6 × 10⁻²² | **0,34** |
| White (p) | non calculé | **0,83** |

**L'hétéroscédasticité disparaît quand on ajoute l'interaction.** Elle n'était pas une propriété
de la cible mais un symptôme de mauvaise spécification : le modèle sans interaction se trompait
systématiquement sur les fumeurs à IMC élevé, et cette erreur structurée se lisait comme une
variance non constante. Corriger la forme fonctionnelle a réglé les deux à la fois.

Conséquence pratique : les erreurs standard robustes (HC3) ne changent presque rien ici, les
écarts allant de 0,88 à 1,20 fois les erreurs classiques. Le remède habituel n'a pas lieu d'être
appliqué.

Les prédictions ne sont pas négatives non plus (minimum 1359 $ sur 1338 observations), alors
que rien dans l'OLS ne le garantit : défaut réel du modèle linéaire sur des données de coûts,
qui ne se manifeste simplement pas sur ce jeu.

## Un GLM Gamma ferait-il mieux ? Non, mesuré

L'argument théorique est solide : les coûts de santé sont strictement positifs et asymétriques,
et la littérature d'économie de la santé recommande un **GLM Gamma à lien logarithmique**
(Jones 2010, Manning & Mullahy 2001). Le modèle relie `log(E[y|x])` au prédicteur linéaire, avec
une variance proportionnelle au carré de la moyenne, soit un coefficient de variation constant.

Deux mesures contredisent l'application de cette recommandation ici.

**Le test de Park modifié ne désigne pas la Gamma.** En régressant `ln((y − ŷ)²)` sur `ln(ŷ)`, la
pente estime l'exposant de la fonction de variance : 0 pour une gaussienne, 1 pour Poisson,
2 pour Gamma. Mesure obtenue : **0,70** (IC95 : 0,57 à 0,84). La structure de variance résiduelle
est plus proche de Poisson que de Gamma, et loin des deux.

**Et en validation croisée, le GLM fait nettement pire :**

| Modèle | MAE (5-fold) |
|---|---|
| **OLS + interaction (le projet)** | **2921 $** |
| GLM Poisson, lien log | 3633 $ |
| GLM Gamma, lien log | 4287 $ |

L'explication tient au critère. Le lien logarithmique optimise une erreur *relative* : se tromper
de 500 $ sur un patient à 2000 $ y pèse autant que 10 000 $ sur un patient à 40 000 $. La MAE, elle,
compte les dollars. Un modèle qui soigne les petits coûts au détriment des gros est pénalisé par
ce critère, et c'est ce qui se produit.

La leçon est plus générale que ce projet : **une famille mieux adaptée à la nature de la variable
n'est pas automatiquement meilleure pour le critère retenu.** Il fallait mesurer.

## Le piège de la retransformation, si l'on passait par le log

Une note pour ne pas confondre deux modèles souvent présentés comme équivalents. Un GLM Gamma à
lien log modélise `log(E[y|x])`, une OLS sur `log(y)` modélise `E[log(y)|x]`. Par l'inégalité de
Jensen, ces deux quantités diffèrent : `E[log y] < log E[y]`.

Exponentier naïvement une prédiction de la seconde **sous-estime systématiquement** la moyenne.
Le correctif classique est le facteur de lissage de Duan (1983), `φ̂ = moyenne(exp(résidus))`, qui
vaut typiquement entre 1,5 et 4 sur des données de coûts de santé (Jones 2010) : un écart qui
dépasserait de très loin le seuil de 3500 $ de l'exercice. Et ce correctif suppose lui-même
l'homoscédasticité sur l'échelle log, faute de quoi le biais dépend de `x`.

Le GLM à lien log évite tout cela : il prédit directement sur l'échelle des dollars, sans
retransformation.

## Pourquoi la validation croisée tranche mieux que le R²

Le R² **augmente mécaniquement** quand on ajoute une variable, même sans rapport avec la cible :
la somme des carrés résiduels ne peut pas croître quand on élargit l'espace du modèle (ISLR
§3.2.2). Ici il passe de 0,751 à 0,841 avec l'interaction, mais cette progression, seule, ne
prouve rien.

Les correctifs de la tradition statistique sont le R² ajusté, le test F emboîté, l'AIC et le BIC.
Le test F sur ce projet donne, en version robuste (test de Wald avec covariance HC3, valide même
sous hétéroscédasticité contrairement au test F classique) : statistique 521, p ≈ 2 × 10⁻¹¹⁵.

Mais ces critères se calculent tous **sur les données d'entraînement**. La validation croisée,
elle, mesure directement l'erreur de généralisation : 4203 $ → 2921 $. C'est un argument plus fort,
et c'est la raison pour laquelle ce projet s'appuie dessus plutôt que sur le R².

Le rapprochement à faire est celui-ci : la tradition statistique pénalise la complexité par une
formule (AIC, BIC, R² ajusté), la tradition machine learning la sanctionne par la mesure sur des
données non vues. Les deux répondent à la même question, la seconde avec moins d'hypothèses.

## Terminologie : deux traditions, un même objet

Ce projet écrit en français avec des outils anglophones, et les deux traditions ne nomment pas
les choses pareil :

| Concept | Tradition statistique | Tradition machine learning |
|---|---|---|
| Variables explicatives | régresseurs, covariables | *features* |
| Variable expliquée | réponse, variable dépendante | *target* |
| Coefficients | paramètres β | *weights*, `coef_` |
| Constante | intercept, terme constant | `intercept_`, *bias* |
| Codage catégoriel | codage disjonctif, variables indicatrices | *one-hot encoding* |
| Estimation | estimation | *fit*, entraînement |
| Pénalisation L2 | régression ridge | *weight decay* |

Deux pièges à connaître :

**« GLM » est ambigu en anglais.** Le *general linear model* (modèle linéaire général : ANOVA,
ANCOVA, régression multivariée gaussienne) et le *generalized linear model* (modèle linéaire
**généralisé** : lien + famille exponentielle) portent presque le même nom. Le français distingue
mieux « général » de « généralisé ».

**`drop_first=True` relève de la tradition statistique.** Elle impose de supprimer une modalité
pour éviter la colinéarité parfaite, sans quoi les coefficients ne sont pas identifiables. La
tradition ML garde souvent toutes les modalités, la régularisation ou le pseudo-inverse rendant le
problème soluble. Sans pénalisation, les *prédictions* seraient identiques dans les deux cas :
seule l'interprétabilité des coefficients change.

## Résumé de la nomenclature

- **Famille** : modèle linéaire (linéaire en les paramètres)
- **Type** : régression linéaire multiple
- **Estimation** : moindres carrés ordinaires (OLS/MCO), sans pénalisation
- **Variables catégorielles** : indicatrices one-hot, une modalité de référence supprimée
- **Non-linéarité** : un terme d'interaction, construit à la main, effets principaux conservés
  (principe hiérarchique)
- **Homoscédasticité** : hypothèse de Gauss-Markov, **vérifiée tenue** une fois l'interaction
  ajoutée (Breusch-Pagan p = 0,34)
- **Normalité des résidus** : jamais requise pour l'estimation, non tenue ici (asymétrie 2,53
  sur l'ensemble du jeu, 2,64 sur le seul jeu de test), sans conséquence sur la MAE mesurée

## Sources

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, 2e éd., §3.2.2
  (R² et statistique F) et §3.3.2 (interactions, principe hiérarchique) :
  <https://www.statlearning.com/>
- Wooldridge, *Introductory Econometrics: A Modern Approach*, ch. 3 (hypothèses MLR.1–MLR.6,
  théorème de Gauss-Markov) et ch. 8 (inférence robuste à l'hétéroscédasticité)
- Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, §3.6, qui
  classent la normalité des résidus comme la **moins importante** des hypothèses
- Jones, A.M. (2010), *Models For Health Care*, HEDG Working Paper 10/01, University of York :
  <https://www.york.ac.uk/media/economics/documents/herc/wp/10_01.pdf> (synthèse sur les GLM
  appliqués aux dépenses de santé, facteur de lissage, test de Park)
- Manning & Mullahy (2001), « Estimating log models: to transform or not to transform? »,
  *Journal of Health Economics* 20(4)
- Duan, N. (1983), « Smearing Estimate: A Nonparametric Retransformation Method », *JASA* 78(383)
- Documentation scikit-learn sur les GLM, qui recommande explicitement « a Gamma distribution
  with a log-link » pour des cibles positives et asymétriques :
  <https://scikit-learn.org/stable/modules/linear_model.html>
