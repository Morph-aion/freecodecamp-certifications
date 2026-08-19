# Mean-Variance-Standard Deviation Calculator

https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/mean-variance-standard-deviation-calculator

## Objectif

Créer une fonction `calculate()` qui utilise NumPy pour calculer la moyenne, variance,
écart-type, max, min et somme d'une matrice 3×3 le long des axes et pour la matrice
aplatie.

## Spécifications

### Entrée
- Une liste de 9 chiffres

### Sortie
- Un dictionnaire contenant 6 clés, chacune contenant 3 valeurs :
  - `mean` : moyenne de chaque axe + matrice aplatie
  - `variance` : variance de chaque axe + matrice aplatie
  - `standard deviation` : écart-type de chaque axe + matrice aplatie
  - `max` : maximum de chaque axe + matrice aplatie
  - `min` : minimum de chaque axe + matrice aplatie
  - `sum` : somme de chaque axe + matrice aplatie

### Structure du dictionnaire

```python
{
  'mean': [axe0, axe1, flatten],
  'variance': [axe0, axe1, flatten],
  'standard deviation': [axe0, axe1, flatten],
  'max': [axe0, axe1, flatten],
  'min': [axe0, axe1, flatten],
  'sum': [axe0, axe1, flatten]
}
```

### Contraintes
- Si la liste ne contient pas exactement 9 éléments, lever un `ValueError` avec le message :
  `"List must contain nine numbers."`
- Utiliser `numpy.array()` pour convertir la liste en matrice
- Utiliser `axis=0` pour les colonnes, `axis=1` pour les lignes

## Lancement

```bash
uv run test_units.py              # tests unitaires
uv run --with marimo marimo edit --sandbox notebook.py   # notebook Marimo
```

Depuis la racine de la certification :

```bash
make test PROJET=01-mean-variance-standard-deviation-calculator
```

`test_module.py`, cité par l'énoncé, est le fichier de correction fourni par
freeCodeCamp dans le boilerplate : il n'est pas versionné ici. `test_units.py`
le couvre et va plus loin, notamment sur le choix du diviseur de la variance.

## Fichiers

| Fichier | Rôle |
|---|---|
| `mean_var_std.py` | La fonction `calculate()` demandée par l'énoncé |
| `test_units.py` | Tests unitaires, dont les deux jeux du corrigé officiel |
| `notebook.py` | Notebook Marimo : exemple commenté et cellule de test officielle |
| `docs/enonce-freecodecamp.md` | Énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions travaillées, expliquées |

## Le piège du projet : quelle variance ?

`numpy.var()` divise par `n` (variance de population), `pandas.var()` par `n − 1`
(variance d'échantillon). Sur la ligne `[0, 1, 2]`, cela fait 0,667 contre 1,0.

Le correcteur attend les valeurs NumPy par défaut. Détail dans
[docs/notions-mobilisees.md](docs/notions-mobilisees.md) §5.

## État : conforme au corrigé officiel

Les deux jeux de `test_module.py` sont reproduits à l'identique, ainsi que le
`ValueError` sur une liste de longueur invalide. 8 tests unitaires.

URL de soumission : *à compléter une fois le projet soumis.*
