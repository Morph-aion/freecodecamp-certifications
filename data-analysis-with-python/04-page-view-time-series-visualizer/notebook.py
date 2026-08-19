# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "marimo>=0.23.3",
# ]
# ///

# Notebook du projet 04 : Page View Time Series Visualizer.
# Aucune logique ici : tout vient de time_series_visualizer.py (convention du projet 01).

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from time_series_visualizer import (
        draw_bar_plot,
        draw_box_plot,
        draw_line_plot,
        load_data,
    )

    return (
        draw_bar_plot,
        draw_box_plot,
        draw_line_plot,
        load_data,
        mo,
        plt,
    )


@app.cell
def _(load_data, mo):
    df = load_data()

    mo.md(
        f"""
        # Page View Time Series Visualizer

        Visualiser les **vues quotidiennes** du forum freeCodeCamp
        (5/2016 – 12/2019). **{len(df)}** jours après filtrage des 2.5% extrêmes.

        Trois visualisations : line plot, bar chart, box plots.
        """
    )
    return (df,)


@app.cell
def _(draw_line_plot, df, mo, plt):
    _fig_line = draw_line_plot(df)
    mo.mpl.interactive(_fig_line)
    return


@app.cell
def _(draw_bar_plot, df, mo, plt):
    _fig_bar = draw_bar_plot(df)
    mo.mpl.interactive(_fig_bar)
    return


@app.cell
def _(draw_box_plot, df, mo, plt):
    _fig_box = draw_box_plot(df)
    mo.mpl.interactive(_fig_box)
    return
