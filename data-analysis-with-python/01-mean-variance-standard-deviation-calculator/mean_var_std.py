# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""Calculateur de moyenne, variance, écart-type, max, min et somme.

Crée une fonction `calculate()` qui prend une liste de 9 chiffres, la convertit
en matrice 3×3 NumPy, et retourne un dictionnaire contenant ces statistiques
le long des axes 0, 1 et pour la matrice aplatie.
"""

import numpy as np


def calculate(list):  # noqa: A002
    """Renvoie les statistiques d'une matrice 3×3 le long des deux axes.

    Le paramètre s'appelle `list` et masque donc le type natif : c'est la
    signature imposée par le boilerplate freeCodeCamp, que le correcteur appelle
    telle quelle. La renommer serait plus propre mais ferait diverger le
    livrable de ce qui est attendu.
    """
    if len(list) != 9:
        raise ValueError("List must contain nine numbers.")

    arr = np.array(list).reshape(3, 3)

    calculations = {
        "mean": [
            arr.mean(axis=0).tolist(),
            arr.mean(axis=1).tolist(),
            arr.mean().tolist(),
        ],
        "variance": [
            arr.var(axis=0).tolist(),
            arr.var(axis=1).tolist(),
            arr.var().tolist(),
        ],
        "standard deviation": [
            arr.std(axis=0).tolist(),
            arr.std(axis=1).tolist(),
            arr.std().tolist(),
        ],
        "max": [
            arr.max(axis=0).tolist(),
            arr.max(axis=1).tolist(),
            arr.max().tolist(),
        ],
        "min": [
            arr.min(axis=0).tolist(),
            arr.min(axis=1).tolist(),
            arr.min().tolist(),
        ],
        "sum": [
            arr.sum(axis=0).tolist(),
            arr.sum(axis=1).tolist(),
            arr.sum().tolist(),
        ],
    }

    return calculations
