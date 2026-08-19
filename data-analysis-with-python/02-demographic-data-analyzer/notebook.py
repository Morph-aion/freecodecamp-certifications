# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "marimo>=0.23.3",
#     "tabulate",
# ]
# ///

# Notebook du projet 02 : Demographic Data Analyzer.
# Aucune logique ici : tout vient de demographic_data_analyzer.py (convention du projet 01).

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from demographic_data_analyzer import calculate_demographic_data, load_data

    return (calculate_demographic_data, load_data, mo, pd)


@app.cell
def _(load_data, mo):
    df = load_data()

    mo.md(
        f"""
        # Demographic Data Analyzer

        Analyser le jeu de données **Census Income** (UCI) avec Pandas.
        **{len(df)}** observations, **{len(df.columns)}** colonnes.

        | Colonne | Type | Exemples |
        |---|---|---|
        | `age` | numérique | 39, 50, 38… |
        | `workclass` | catégorielle | State-gov, Self-emp-not-inc… |
        | `education` | catégorielle | Bachelors, HS-grad, Masters… |
        | `race` | catégorielle | White, Black, Asian-Pac-Islander… |
        | `sex` | catégorielle | Male, Female |
        | `salary` | **cible** | <=50K, >50K |
        """
    )
    return (df,)


@app.cell
def _(calculate_demographic_data, mo):
    # print_data=False : dans un notebook, la sortie passe par mo.md, pas par
    # les print() du script.
    results = calculate_demographic_data(print_data=False)

    mo.md(
        f"""
        ## 2. Résultats

        ### Répartition par race

        {results["race_count"].to_frame().to_markdown()}

        ---

        ### Statistiques

        | Question | Réponse |
        |---|---|
        | Âge moyen des hommes | **{results["average_age_men"]:.1f}** ans |
        | % Bachelor's degree | **{results["percentage_bachelors"]:.1f}%** |
        | % éducation avancée >50K | **{results["higher_education_rich"]:.1f}%** |
        | % sans éducation avancée >50K | **{results["lower_education_rich"]:.1f}%** |
        | Heures min/semaine | **{results["min_work_hours"]}** |
        | % min heures >50K | **{results["rich_percentage"]:.1f}%** |
        | Pays le plus riche | **{results["highest_earning_country"]}** ({results["highest_earning_country_percentage"]:.1f}%) |
        | Occupation >50K en Inde | **{results["top_IN_occupation"]}** |
        """
    )
    return (results,)
