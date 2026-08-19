# Demographic Data Analyzer

https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer

## Objectif

Analyser le jeu de données de recensement adulte (`adult.data.csv`) avec Pandas pour
répondre à des questions démographiques précises.

## Questions à répondre

1. Combien de personnes de chaque race sont représentées ? (Série Pandas avec noms de races en index)
2. Quel est l'âge moyen des hommes ?
3. Quel est le pourcentage de personnes ayant un diplôme de Bachelor ?
4. Quel pourcentage de personnes avec une éducation avancée (Bachelors, Masters, ou Doctorate) gagnent plus de 50K ?
5. Quel pourcentage de personnes sans éducation avancée gagnent plus de 50K ?
6. Quel est le nombre minimum d'heures travaillées par semaine ?
7. Quel pourcentage des personnes travaillant le minimum d'heures gagnent plus de 50K ?
8. Quel pays a le plus haut pourcentage de personnes gagnant >50K et quel est ce pourcentage ?
9. Quelle est l'occupation la plus populaire pour ceux gagnant >50K en Inde ?

## Structure du fichier de sortie

Le fichier `demographic_data_analyzer.py` doit contenir une fonction `calculate_demographic_data()` retournant un dictionnaire avec les 9 réponses.

## Données

Le fichier `adult.data.csv` est inclus dans le boilerplate. Colonnes :
- `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, `salary`

## Lancement

```bash
uv run demographic_data_analyzer.py     # calcule et affiche les 9 réponses
uv run test_units.py                    # tests unitaires
uv run --with marimo marimo edit --sandbox notebook.py
```

Depuis la racine de la certification :

```bash
make test CERTIF=data-analysis-with-python PROJET=02-demographic-data-analyzer
```

Le script télécharge `adult.data.csv` depuis le dépôt freeCodeCamp au premier
lancement (`data/raw/` est gitignoré).

## Fichiers

| Fichier | Rôle |
|---|---|
| `demographic_data_analyzer.py` | `calculate_demographic_data()` : les 9 réponses |
| `test_units.py` | Tests unitaires, dont les 10 assertions du corrigé officiel |
| `notebook.py` | Notebook Marimo : exploration et résultats commentés |
| `docs/enonce-freecodecamp.md` | Énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions travaillées, expliquées |
| `data/raw/` | Jeu de données brut, non versionné : créé et rempli au premier lancement |

## Le piège du projet : le contrat de sortie

Deux points ne se devinent pas depuis l'énoncé, et font échouer la soumission
sans que le calcul soit faux.

**Les noms de clés sont imposés.** Le correcteur lit `data['average_age_men']`,
pas `average_age_males` ; `percentage_bachelors`, pas `bachelor_percentage` ;
`higher_education_rich`, `lower_education_rich`, `rich_percentage`. Ces noms
viennent de `test_module.py`, pas de l'énoncé.

**La signature aussi.** Le correcteur appelle
`calculate_demographic_data(print_data=False)` : sans ce paramètre, l'appel
lève un `TypeError` avant toute vérification.

## L'autre piège : le CSV a déjà un en-tête

`adult.data.csv` porte sa ligne de titres. Passer `header=None` avec des noms
explicites, réflexe naturel pour ce jeu de données dont la version UCI d'origine
n'a pas d'en-tête, transforme cette ligne en observation : 32562 lignes au lieu
de 32561, et les colonnes numériques basculent en texte. La première moyenne
échoue alors sur `Cannot perform reduction 'mean' with string dtype`.

`load_data()` vérifie désormais que les colonnes lues correspondent à celles
attendues, plutôt que de les imposer.

## État : conforme au corrigé officiel

Les 10 assertions de `test_module.py` sont reproduites et vérifiées :
`[27816, 3124, 1039, 311, 271]`, 39,4, 16,4 %, 46,5 %, 17,4 %, 1 h, 10 %,
Iran 41,9 %, Prof-specialty. 21 tests unitaires.

URL de soumission : *à compléter une fois le projet soumis.*
