# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Prédiction de l'élévation du niveau de la mer avec régression linéaire.

Analyse les données EPA depuis 1880 et prédit l'élévation jusqu'en 2050
avec deux lignes de tendance : toutes les données, et depuis 2000.
"""

import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures"
DATA_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/"
    "boilerplate-sea-level-predictor/main/epa-sea-level.csv"
)


def download_data(csv_path: pathlib.Path) -> None:
    """Télécharge epa-sea-level.csv depuis freeCodeCamp si absent."""
    print(f"Téléchargement depuis {DATA_URL}…")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            csv_path.write_bytes(response.read())
    except OSError as err:
        # Arrêt net : sans cela, `load_data()` enchaîne sur la lecture d'un
        # fichier absent et l'erreur remontée ne dit rien de la cause.
        print(f"Erreur : téléchargement impossible ({err})")
        print(f"Télécharger manuellement {DATA_URL} vers {csv_path}")
        sys.exit(1)


def load_data() -> pd.DataFrame:
    """Charge le jeu de données, en le téléchargeant si absent."""
    csv_path = DATA_DIR / "epa-sea-level.csv"
    if not csv_path.exists():
        download_data(csv_path)
    return pd.read_csv(csv_path)


def draw_plot(df: pd.DataFrame | None = None) -> plt.Axes:
    """Crée le scatter plot avec deux lignes de tendance.

    Renvoie l'**Axes**, pas la Figure : le correcteur fait
    `ax = sea_level_predictor.draw_plot()` puis lit directement `ax.get_title()`
    et `ax.get_lines()`. C'est une divergence avec les projets 03 et 04, dont
    les fonctions renvoient une Figure.

    L'argument est optionnel, le correcteur appelant `draw_plot()` sans rien.
    """
    if df is None:
        df = load_data()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot des données brutes
    ax.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], alpha=0.5, label="Raw Data")

    # Ligne de tendance 1 : toutes les données
    slope, intercept, *_ = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    years_all = range(int(df["Year"].min()), 2051)
    ax.plot(
        years_all,
        [slope * y + intercept for y in years_all],
        "r",
        label="Best Fit Line (All Data)",
    )

    # Ligne de tendance 2 : depuis 2000
    df_recent = df[df["Year"] >= 2000]
    slope_r, intercept_r, *_ = linregress(
        df_recent["Year"], df_recent["CSIRO Adjusted Sea Level"]
    )
    years_recent = range(2000, 2051)
    ax.plot(
        years_recent,
        [slope_r * y + intercept_r for y in years_recent],
        "g",
        label="Best Fit Line (2000-Present)",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Sea Level (inches)", fontsize=12)
    ax.set_title("Rise in Sea Level", fontsize=16)
    ax.legend()
    fig.tight_layout()
    return ax


def main() -> None:
    """Pipeline complet : chargement, visualisation."""
    print("Chargement des données…")
    df = load_data()
    print(f"  {len(df)} lignes")

    print("Prédiction…")
    slope_all, intercept_all, *_ = linregress(
        df["Year"], df["CSIRO Adjusted Sea Level"]
    )
    df_2000 = df[df["Year"] >= 2000]
    slope_2000, intercept_2000, *_ = linregress(
        df_2000["Year"], df_2000["CSIRO Adjusted Sea Level"]
    )
    print(f"  Tendance globale : {slope_all:.4f} pouces/an")
    print(f"  Tendance depuis 2000 : {slope_2000:.4f} pouces/an")
    print(f"  Prédiction 2050 (global) : {slope_all * 2050 + intercept_all:.2f} pouces")
    print(
        f"  Prédiction 2050 (depuis 2000) : {slope_2000 * 2050 + intercept_2000:.2f} pouces"
    )

    print("Graphique…")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax = draw_plot(df)
    ax.figure.savefig(FIGURES_DIR / "01-sea-level.png", dpi=150)
    plt.close(ax.figure)
    print(f"  Sauvegardé : {FIGURES_DIR / '01-sea-level.png'}")

    print("Terminé.")


if __name__ == "__main__":
    main()
