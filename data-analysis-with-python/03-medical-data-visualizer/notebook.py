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

# Notebook du projet 03 : Medical Data Visualizer.
# Aucune logique ici : tout vient de medical_data_visualizer.py (convention du projet 01).

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from medical_data_visualizer import (
        add_overweight,
        clean_data,
        draw_cat_plot,
        draw_heat_map,
        load_data,
        normalize_values,
    )

    return (
        add_overweight,
        clean_data,
        draw_cat_plot,
        draw_heat_map,
        load_data,
        mo,
        normalize_values,
        plt,
    )


@app.cell
def _(load_data, mo):
    df_raw = load_data()

    mo.md(
        f"""
        # Medical Data Visualizer

        Visualiser des **données d'examen médical** avec Matplotlib et Seaborn.
        **{len(df_raw)}** observations, **{len(df_raw.columns)}** colonnes.

        | Colonne | Type | Description |
        |---|---|---|
        | `age` | numérique | Âge en jours |
        | `sex` | catégorielle | 1 = femme, 2 = homme (11 lignes portent 3) |
        | `height` | numérique | Taille en cm |
        | `weight` | numérique | Poids en kg |
        | `ap_hi` / `ap_lo` | numérique | Pression systolique / diastolique |
        | `cholesterol` | ordinal | 1 = normal, 2, 3 = élevé |
        | `gluc` | ordinal | 1 = normal, 2, 3 = élevé |
        | `smoke`, `alco`, `active` | binaire | Habitudes de vie |
        | `cardio` | **cible** | 0 = pas de maladie, 1 = maladie cardiaque |
        """
    )
    return (df_raw,)


@app.cell
def _(add_overweight, clean_data, df_raw, mo, normalize_values):
    df = clean_data(df_raw)
    df = add_overweight(df)
    df = normalize_values(df)

    mo.md(
        f"""
        ## 2. Nettoyage et features

        Après nettoyage : **{len(df)}** lignes (pression valide, percentiles).

        | Feature | Raison |
        |---|---|
        | `overweight` | IMC > 25 → 1, sinon 0 |
        | `cholesterol` | Binarisé : 1 → 0 (normal), 2/3 → 1 (anormal) |
        | `gluc` | Binarisé : 1 → 0 (normal), 2/3 → 1 (anormal) |
        """
    )
    return (df,)


@app.cell
def _(draw_cat_plot, df, mo, plt):
    _fig_cat = draw_cat_plot(df)
    mo.mpl.interactive(_fig_cat)
    return


@app.cell
def _(draw_heat_map, df, mo, plt):
    _fig_heat = draw_heat_map(df)
    mo.mpl.interactive(_fig_heat)
    return
