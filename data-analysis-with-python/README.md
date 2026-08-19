# Data Analysis with Python

https://www.freecodecamp.org/learn/data-analysis-with-python/

Certification validée par les 5 projets ci-dessous (tests automatiques). Chaque projet
applique une facette de l'analyse de données : calculs NumPy, exploration Pandas,
visualisation Matplotlib/Seaborn, séries temporelles, et régression linéaire.

## Projets

| # | Projet | Nature | Statut | Dossier |
|---|---|---|---|---|
| 1 | Mean-Variance-Standard Deviation Calculator | Calculs statistiques (NumPy) | Défi réussi, à soumettre | [01-mean-variance-standard-deviation-calculator/](01-mean-variance-standard-deviation-calculator/) |
| 2 | Demographic Data Analyzer | Exploration de données (Pandas) | Défi réussi, à soumettre | [02-demographic-data-analyzer/](02-demographic-data-analyzer/) |
| 3 | Medical Data Visualizer | Visualisation (Matplotlib/Seaborn) | Défi réussi, à soumettre | [03-medical-data-visualizer/](03-medical-data-visualizer/) |
| 4 | Page View Time Series Visualizer | Séries temporelles (Pandas/Matplotlib) | Défi réussi, à soumettre | [04-page-view-time-series-visualizer/](04-page-view-time-series-visualizer/) |
| 5 | Sea Level Predictor | Régression linéaire (SciPy) | Défi réussi, à soumettre | [05-sea-level-predictor/](05-sea-level-predictor/) |

## Processus de soumission

Contrairement à la certification Machine Learning, ces cinq projets ne passent
pas par un notebook : chacun a un **boilerplate GitHub officiel** contenant un
fichier `.py` à compléter, `main.py` pour l'exécuter et `test_module.py` pour le
vérifier.

1. Ouvrir le code de départ via le lien de la page du projet
2. Écrire la solution dans le fichier `.py` attendu (`mean_var_std.py`, etc.)
3. Passer les tests automatiques (`python3 main.py`)
4. Soumettre l'URL du projet via « I've completed this challenge »

Les modules de ce dépôt portent déjà les noms attendus par les correcteurs : le
contenu se colle tel quel, à l'en-tête PEP 723 près, qui ne sert qu'à
l'exécution locale par `uv run`.

Les boilerplates GitHub officiels :
- `freeCodeCamp/boilerplate-mean-variance-standard-deviation-calculator`
- `freeCodeCamp/boilerplate-demographic-data-analyzer`
- `freeCodeCamp/boilerplate-medical-data-visualizer`
- `freeCodeCamp/boilerplate-page-view-time-series-visualizer`
- `freeCodeCamp/boilerplate-sea-level-predictor`

## Stack technique

| Bibliothèque | Usage |
|---|---|
| NumPy | Calculs statistiques, manipulations de tableaux |
| Pandas | Exploration, filtrage, groupby, manipulation de DataFrames |
| Matplotlib | Graphiques line, bar, scatter, box |
| Seaborn | Heatmaps, box plots statistiques |
| SciPy | Régression linéaire (`linregress`) |

## Format de travail

- Fichier `.py` à compléter dans le boilerplate officiel, pas de notebook
- Code Python standalone pour le développement local
- Chaque projet README documente l'URL de soumission finale une fois complété
