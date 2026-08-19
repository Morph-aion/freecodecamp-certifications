# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Visualisation de données médicales avec Matplotlib et Seaborn.

Crée deux visualisations :
1. Un graphique catégoriel (count) pour les variables de santé par statut cardio
2. Une matrice de corrélation sous forme de heatmap
"""

import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures"
DATA_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/"
    "boilerplate-medical-data-visualizer/main/medical_examination.csv"
)


def download_data(csv_path: pathlib.Path) -> None:
    """Télécharge medical_examination.csv depuis freeCodeCamp si absent."""
    print(f"Téléchargement depuis {DATA_URL}…")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            csv_path.write_bytes(response.read())
    except OSError as err:
        # Arrêt net plutôt que retour silencieux : sans cela, `load_data()`
        # enchaîne sur un `read_csv` d'un fichier absent et l'utilisateur reçoit
        # un FileNotFoundError qui ne dit rien de la cause réelle.
        print(f"Erreur : téléchargement impossible ({err})")
        print(f"Télécharger manuellement {DATA_URL} vers {csv_path}")
        sys.exit(1)


def load_data() -> pd.DataFrame:
    """Charge le jeu de données, en le téléchargeant si absent."""
    csv_path = DATA_DIR / "medical_examination.csv"
    if not csv_path.exists():
        download_data(csv_path)
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données : supprime les lignes invalides.

    Filtres appliqués :
    - ap_lo <= ap_hi (diastolique <= systolique)
    - height entre 2.5ᵉ et 97.5ᵉ percentile
    - weight entre 2.5ᵉ et 97.5ᵉ percentile
    """
    # Les quatre bornes sont calculées sur le jeu complet, puis appliquées en
    # une seule passe. Filtrer séquentiellement en recalculant les quantiles à
    # chaque étape rétrécit la distribution au fur et à mesure et retire 475
    # lignes de trop (62 784 au lieu de 63 259), ce qui décale les corrélations
    # de la heatmap au dixième près.
    h_low, h_high = df["height"].quantile([0.025, 0.975])
    w_low, w_high = df["weight"].quantile([0.025, 0.975])

    return df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= h_low)
        & (df["height"] <= h_high)
        & (df["weight"] >= w_low)
        & (df["weight"] <= w_high)
    ].copy()


def add_overweight(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne 'overweight' (IMC > 25)."""
    df = df.copy()
    df["overweight"] = (df["weight"] / ((df["height"] / 100) ** 2) > 25).astype(int)
    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise cholesterol et gluc : 1 → 0 (normal), >1 → 1 (anormal)."""
    df = df.copy()
    df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
    df["gluc"] = (df["gluc"] > 1).astype(int)
    return df


def draw_cat_plot(df: pd.DataFrame | None = None) -> plt.Figure:
    """Crée le graphique catégoriel de comptage par statut cardio.

    `df` est optionnel : le correcteur appelle `draw_cat_plot()` sans argument
    et attend que la fonction charge et prépare les données elle-même.

    Trois contraintes viennent de `test_module.py` plutôt que de l'énoncé :
    `overweight` fait partie des variables tracées, l'axe des abscisses porte
    les noms de variables (pas les valeurs 0/1), et l'axe des ordonnées
    s'intitule « total ».
    """
    if df is None:
        df = prepare_data()

    cat_cols = ["active", "alco", "cholesterol", "gluc", "overweight", "smoke"]
    df_cat = df.melt(
        id_vars=["cardio"],
        value_vars=cat_cols,
        var_name="variable",
        value_name="value",
    )
    # Comptage explicite plutôt que kind="count" : celui-ci compterait les
    # occurrences de `x`, alors qu'il faut une barre par couple (variable,
    # valeur), soit six variables × deux valeurs par panneau.
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"], observed=True)
        .size()
        .reset_index(name="total")
    )

    g = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    )
    return g.figure


def prepare_data() -> pd.DataFrame:
    """Charge, nettoie et enrichit le jeu de données.

    Regroupe la chaîne complète pour que `draw_cat_plot()` et `draw_heat_map()`
    puissent être appelées sans argument, comme le fait le correcteur.
    """
    return normalize_values(add_overweight(clean_data(load_data())))


def draw_heat_map(df: pd.DataFrame | None = None) -> plt.Figure:
    """Crée la heatmap de corrélation avec triangle supérieur masqué.

    `df` est optionnel, pour la même raison que `draw_cat_plot()`.
    """
    if df is None:
        df = prepare_data()

    corr = df.corr()
    # Masque du triangle supérieur repris de l'exemple officiel seaborn
    # « Plotting a diagonal correlation matrix » :
    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    # L'énoncé freeCodeCamp reprend lui-même les étapes de cet exemple, jusqu'aux
    # noms `corr` et `mask` pré-déclarés dans le fichier de départ.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Pipeline complet : chargement, nettoyage, visualisation."""
    print("Chargement des données…")
    df = load_data()
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")

    print("Nettoyage…")
    df = clean_data(df)
    print(f"  {len(df)} lignes après nettoyage")

    print("Ajout de 'overweight'…")
    df = add_overweight(df)

    print("Normalisation de cholesterol et gluc…")
    df = normalize_values(df)

    print("Graphique catégoriel…")
    fig_cat = draw_cat_plot(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_cat.savefig(FIGURES_DIR / "01-cat-plot.png", dpi=150)
    plt.close(fig_cat)
    print(f"  Sauvegardé : {FIGURES_DIR / '01-cat-plot.png'}")

    print("Heatmap de corrélation…")
    fig_heat = draw_heat_map(df)
    fig_heat.savefig(FIGURES_DIR / "02-heatmap.png", dpi=150)
    plt.close(fig_heat)
    print(f"  Sauvegardé : {FIGURES_DIR / '02-heatmap.png'}")

    print("Terminé.")


if __name__ == "__main__":
    main()
