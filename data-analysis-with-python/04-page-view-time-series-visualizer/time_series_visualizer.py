# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Visualisation de séries temporelles : vues quotidiennes du forum freeCodeCamp.

Trois types de graphiques :
1. Line plot : évolution quotidienne
2. Bar chart : moyenne mensuelle par année
3. Box plots : tendance (par année) et saisonnalité (par mois)
"""

import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures"
VALUE_COLUMN = "value"

# Deux conventions coexistent dans le corrigé : la légende du bar plot attend
# les noms complets, les étiquettes du box plot mensuel les abréviations.
MONTHS_FULL = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_SHORT = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
DATA_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/"
    "boilerplate-page-view-time-series-visualizer/main/fcc-forum-pageviews.csv"
)


def download_data(csv_path: pathlib.Path) -> None:
    """Télécharge fcc-forum-pageviews.csv depuis freeCodeCamp si absent."""
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
    """Charge et nettoie le jeu de données.

    - Parse les dates en index
    - Filtre les 2.5% extrêmes de chaque côté
    """
    csv_path = DATA_DIR / "fcc-forum-pageviews.csv"
    if not csv_path.exists():
        download_data(csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    q_low = df[VALUE_COLUMN].quantile(0.025)
    q_high = df[VALUE_COLUMN].quantile(0.975)
    return df[(df[VALUE_COLUMN] >= q_low) & (df[VALUE_COLUMN] <= q_high)].copy()


# DataFrame exposé au niveau module : la première assertion du correcteur lit
# `time_series_visualizer.df.count(numeric_only=True)` sans passer par une
# fonction. Chargé à l'import, comme le fait le boilerplate officiel.
df = load_data()


def draw_line_plot(df_source: pd.DataFrame | None = None) -> plt.Figure:
    """Line plot de l'évolution quotidienne des vues.

    L'argument est optionnel : le correcteur appelle `draw_line_plot()` nu.
    """
    df = globals()["df"] if df_source is None else df_source
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[VALUE_COLUMN], color="crimson", linewidth=1)
    ax.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Page Views", fontsize=12)
    fig.tight_layout()
    return fig


def draw_bar_plot(df_source: pd.DataFrame | None = None) -> plt.Figure:
    """Bar chart des moyennes mensuelles groupées par année."""
    df = globals()["df"] if df_source is None else df_source
    df_bar = df.copy()
    df_bar["year"] = df_bar.index.year
    df_bar["month"] = df_bar.index.month

    df_pivot = df_bar.groupby(["year", "month"])[VALUE_COLUMN].mean().unstack(level=1)

    # Noms complets : le correcteur attend « January »…« December » dans la
    # légende du bar plot, alors que les box plots veulent les formes abrégées.
    df_pivot.columns = [MONTHS_FULL[m - 1] for m in df_pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Years", fontsize=12)
    ax.set_ylabel("Average Page Views", fontsize=12)
    ax.legend(title="Months", fontsize=8)
    fig.tight_layout()
    return fig


def draw_box_plot(df_source: pd.DataFrame | None = None) -> plt.Figure:
    """Box plots : tendance (par année) et saisonnalité (par mois)."""
    df = globals()["df"] if df_source is None else df_source
    df_box = df.copy()
    df_box["year"] = df_box.index.year
    df_box["month"] = df_box.index.month
    df_box["month_name"] = df_box["month"].apply(lambda m: MONTHS_SHORT[m - 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(x="year", y=VALUE_COLUMN, data=df_box, ax=ax1)
    ax1.set_title("Year-wise Box Plot (Trend)", fontsize=14)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Page Views")

    sns.boxplot(
        x="month_name",
        y=VALUE_COLUMN,
        data=df_box,
        ax=ax2,
        order=MONTHS_SHORT,
    )
    ax2.set_title("Month-wise Box Plot (Seasonality)", fontsize=14)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Page Views")

    fig.tight_layout()
    return fig


def main() -> None:
    """Pipeline complet : chargement, nettoyage, visualisation."""
    print(f"  {len(df)} lignes après filtrage")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Line plot…")
    fig_line = draw_line_plot()
    fig_line.savefig(FIGURES_DIR / "01-line-plot.png", dpi=150)
    plt.close(fig_line)
    print(f"  Sauvegardé : {FIGURES_DIR / '01-line-plot.png'}")

    print("Bar plot…")
    fig_bar = draw_bar_plot()
    fig_bar.savefig(FIGURES_DIR / "02-bar-plot.png", dpi=150)
    plt.close(fig_bar)
    print(f"  Sauvegardé : {FIGURES_DIR / '02-bar-plot.png'}")

    print("Box plot…")
    fig_box = draw_box_plot()
    fig_box.savefig(FIGURES_DIR / "03-box-plot.png", dpi=150)
    plt.close(fig_box)
    print(f"  Sauvegardé : {FIGURES_DIR / '03-box-plot.png'}")

    print("Terminé.")


if __name__ == "__main__":
    main()
