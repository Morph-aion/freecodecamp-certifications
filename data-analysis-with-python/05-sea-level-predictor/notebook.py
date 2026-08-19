# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "marimo>=0.23.3",
# ]
# ///

# Notebook du projet 05 : Sea Level Predictor.
# Aucune logique ici : tout vient de sea_level_predictor.py (convention du projet 01).

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from sea_level_predictor import draw_plot, load_data

    return (draw_plot, load_data, mo, plt)


@app.cell
def _(load_data, mo):
    df = load_data()

    mo.md(
        f"""
        # Sea Level Predictor

        Analyser l'**élévation du niveau de la mer** depuis 1880 et prédire
        jusqu'en 2050 avec la **régression linéaire** (SciPy `linregress`).
        **{len(df)}** mesures.
        """
    )
    return (df,)


@app.cell
def _(draw_plot, df, mo, plt):
    # draw_plot renvoie un Axes (contrat du correcteur) : mo.mpl.interactive
    # attend une Figure, d'où le passage par .figure.
    _ax = draw_plot(df)
    mo.mpl.interactive(_ax.figure)
    return
