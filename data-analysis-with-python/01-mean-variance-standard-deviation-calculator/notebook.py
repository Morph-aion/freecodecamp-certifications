# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "marimo>=0.23.3",
# ]
# ///

# Notebook du projet 01 : Mean-Variance-Standard Deviation Calculator.
# Aucune logique ici : tout vient de mean_var_std.py (convention du projet 01).

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from mean_var_std import calculate

    return (calculate, mo)


@app.cell
def _(calculate, mo):
    mo.md(
        """
        # Mean-Variance-Standard Deviation Calculator

        Calculer la **moyenne**, **variance**, **écart-type**, **max**, **min** et
        **somme** d'une matrice 3×3 le long des axes et pour la matrice aplatie.

        **Entrée** : liste de 9 chiffres.
        **Sortie** : dictionnaire avec 6 clés, chacune contenant 3 valeurs.
        """
    )
    return


@app.cell
def _(calculate, mo):
    _exemple = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    _result = calculate(_exemple)

    mo.md(
        f"""
        ## 2. Exemple avec `{{0, 1, 2, 3, 4, 5, 6, 7, 8}}`

        **Entrée** : `{_exemple}`

        **Matrice 3×3 :**
        ```
        {_exemple[0]} {_exemple[1]} {_exemple[2]}
        {_exemple[3]} {_exemple[4]} {_exemple[5]}
        {_exemple[6]} {_exemple[7]} {_exemple[8]}
        ```

        **Résultat :**

        | Statistique | Axe 0 (colonnes) | Axe 1 (lignes) | Flatten |
        |---|---|---|---|
        | mean | `{_result["mean"][0]}` | `{_result["mean"][1]}` | `{_result["mean"][2]}` |
        | variance | `{_result["variance"][0]}` | `{_result["variance"][1]}` | `{_result["variance"][2]:.4f}` |
        | std | `{_result["standard deviation"][0][0]:.4f}` | `{_result["standard deviation"][1][0]:.4f}` | `{_result["standard deviation"][2]:.4f}` |
        | max | `{_result["max"][0]}` | `{_result["max"][1]}` | `{_result["max"][2]}` |
        | min | `{_result["min"][0]}` | `{_result["min"][1]}` | `{_result["min"][2]}` |
        | sum | `{_result["sum"][0]}` | `{_result["sum"][1]}` | `{_result["sum"][2]}` |
        """
    )
    return


@app.cell
def _(calculate, mo):
    # Les deux jeux du corrigé officiel (test_module.py du boilerplate), repris
    # tels quels plutôt qu'une valeur inventée : c'est ce que le correcteur
    # exécute. La suite [0..8] est trompeuse pour vérifier une implémentation,
    # ses colonnes étant régulières.
    _cas = [
        (
            [2, 6, 2, 8, 4, 0, 1, 5, 7],
            {
                "mean": [
                    [3.6666666666666665, 5.0, 3.0],
                    [3.3333333333333335, 4.0, 4.333333333333333],
                    3.888888888888889,
                ],
                "variance": [
                    [9.555555555555557, 0.6666666666666666, 8.666666666666666],
                    [3.555555555555556, 10.666666666666666, 6.222222222222221],
                    6.987654320987654,
                ],
                "standard deviation": [
                    [3.091206165165235, 0.816496580927726, 2.943920288775949],
                    [1.8856180831641267, 3.265986323710904, 2.494438257849294],
                    2.6434171674156266,
                ],
                "max": [[8, 6, 7], [6, 8, 7], 8],
                "min": [[1, 4, 0], [2, 0, 1], 0],
                "sum": [[11, 15, 9], [10, 12, 13], 35],
            },
        ),
        (
            [9, 1, 5, 3, 3, 3, 2, 9, 0],
            {
                "mean": [
                    [4.666666666666667, 4.333333333333333, 2.6666666666666665],
                    [5.0, 3.0, 3.6666666666666665],
                    3.888888888888889,
                ],
                "variance": [
                    [9.555555555555555, 11.555555555555557, 4.222222222222222],
                    [10.666666666666666, 0.0, 14.888888888888891],
                    9.209876543209875,
                ],
                "standard deviation": [
                    [3.0912061651652345, 3.39934634239519, 2.0548046676563256],
                    [3.265986323710904, 0.0, 3.8586123009300755],
                    3.0347778408328137,
                ],
                "max": [[9, 9, 5], [9, 3, 9], 9],
                "min": [[2, 1, 0], [1, 3, 0], 0],
                "sum": [[14, 13, 8], [15, 9, 11], 35],
            },
        ),
    ]
    _lignes = ""
    _tous = True
    for _entree, _attendu in _cas:
        _ok = calculate(_entree) == _attendu
        _tous = _tous and _ok
        _lignes += f"| `{_entree}` | {'Pass' if _ok else 'Fail'} |\n"

    try:
        calculate([2, 6, 2, 8, 4, 0, 1])
        _leve = False
    except ValueError:
        _leve = True
    _lignes += (
        f"| liste de 7 éléments lève `ValueError` | {'Pass' if _leve else 'Fail'} |\n"
    )

    mo.md(
        f"""
        ## 3. Test officiel freeCodeCamp

        Les trois assertions de `test_module.py`, telles que le correcteur les
        exécute.

        | Cas | Résultat |
        |---|---|
        {_lignes}

        **{"Les trois assertions passent." if _tous and _leve else "Au moins une assertion échoue."}**
        """
    )
    return
