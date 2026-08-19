# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
#     "matplotlib",
# ]
# ///
"""Pipeline complet : chargement, préparation, entraînement, évaluation.

Prédit des coûts de santé à l'aide d'un algorithme de régression linéaire.
Seuil de réussite freeCodeCamp : MAE < 3500.
"""

import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures"
DATA_URL = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"

# freeCodeCamp a renommé la colonne cible « expenses » en « charges » selon les
# versions du jeu de données, sans mettre l'énoncé à jour. Les deux noms sont
# acceptés pour que le code fonctionne aussi bien en local que dans le notebook
# Colab officiel, dont la cellule de test peut référencer l'un ou l'autre.
TARGET_CANDIDATES = ("expenses", "charges")

# Colonnes catégorielles à encoder en indicatrices.
CATEGORICAL_COLUMNS = ("sex", "smoker", "region")


def resolve_target(df: pd.DataFrame) -> str:
    """Renvoie le nom de la colonne cible présente dans le jeu de données."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    raise KeyError(
        f"Aucune colonne cible trouvée parmi {TARGET_CANDIDATES} : {list(df.columns)}"
    )


def download_data(csv_path: pathlib.Path) -> None:
    """Télécharge insurance.csv depuis freeCodeCamp si absent."""
    print(f"Téléchargement du jeu de données depuis {DATA_URL}…")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Le CDN freeCodeCamp renvoie 403 sans User-Agent explicite : celui par
    # défaut d'urllib (« Python-urllib/x.y ») est filtré.
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            csv_path.write_bytes(response.read())
    except OSError as err:
        print(f"Erreur : téléchargement impossible ({err})")
        print(f"Télécharger manuellement {DATA_URL} vers {csv_path}")
        sys.exit(1)


def load_data() -> pd.DataFrame:
    """Charge le jeu de données insurance.csv, en le téléchargeant si absent."""
    csv_path = DATA_DIR / "insurance.csv"
    if not csv_path.exists():
        download_data(csv_path)
    return pd.read_csv(csv_path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les données catégorielles en nombres via one-hot encoding.

    Les colonnes attendues sont nommées explicitement plutôt que détectées par
    dtype : une colonne catégorielle qui apparaîtrait dans une autre version du
    jeu de données serait alors traitée comme numérique sans que rien ne le
    signale. Leur absence lève ici au lieu de passer inaperçue.
    """
    manquantes = [c for c in CATEGORICAL_COLUMNS if c not in df.columns]
    if manquantes:
        raise KeyError(
            f"Colonnes catégorielles absentes : {manquantes}. "
            f"Colonnes disponibles : {list(df.columns)}."
        )
    return pd.get_dummies(df, columns=list(CATEGORICAL_COLUMNS), drop_first=True)


def add_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la feature d'interaction bmi × smoker_yes.

    Échoue explicitement si les colonnes attendues sont absentes : sans cette
    interaction le modèle dépasse le seuil de 3500 (MAE ≈ 4200), et une absence
    silencieuse produirait un échec sans cause visible. Appeler
    prepare_features() en amont.
    """
    manquantes = [c for c in ("bmi", "smoker_yes") if c not in df.columns]
    if manquantes:
        raise KeyError(
            f"Colonnes requises absentes pour l'interaction : {manquantes}. "
            f"Colonnes disponibles : {list(df.columns)}. "
            "prepare_features() doit être appliqué avant add_interaction()."
        )
    df = df.copy()
    df["bmi_smoker"] = df["bmi"] * df["smoker_yes"]
    return df


def split_data(
    df: pd.DataFrame,
    target: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Sépare features et labels, puis split 80/20.

    La cible est détectée automatiquement (« expenses » ou « charges ») si elle
    n'est pas fournie explicitement.
    """
    target = target or resolve_target(df)
    labels = df[target]
    features = df.drop(columns=[target])
    return train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )


def train_model(X_train, y_train) -> LinearRegression:
    """Entraîne un modèle de régression linéaire."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Évalue le modèle et renvoie les métriques, prédictions incluses."""
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def plot_predictions(y_test, y_pred, output_path: pathlib.Path) -> None:
    """Trace le nuage de points prédictions vs réel."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Prédiction parfaite",
    )
    ax.set_xlabel("Dépenses réelles ($)")
    ax.set_ylabel("Dépenses prédites ($)")
    ax.set_title("Prédictions vs Réalités (régression linéaire)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def main() -> None:
    """Pipeline complet."""
    print("Chargement des données…")
    df = load_data()
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")

    print("Préparation des features…")
    target = resolve_target(df)
    df = prepare_features(df)
    # Compté en excluant explicitement la cible : un `len(df.columns) - 1`
    # supposerait qu'elle est toujours présente à cet endroit du pipeline, ce que
    # rien ne garantit si l'ordre des étapes change.
    n_encodage = len(df.columns.drop(target))
    df = add_interaction(df)
    print(
        f"  {n_encodage} features après encodage, "
        f"{len(df.columns.drop(target))} après ajout de l'interaction"
    )

    print("Split 80/20…")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  train : {len(X_train)} lignes, test : {len(X_test)} lignes")

    print("Entraînement…")
    model = train_model(X_train, y_train)

    print("Évaluation…")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  MAE  : {metrics['mae']:.2f} $")
    print(f"  RMSE : {metrics['rmse']:.2f} $")
    print(f"  R²   : {metrics['r2']:.4f}")

    seuil = 3500
    if metrics["mae"] < seuil:
        print(f"\nMAE ({metrics['mae']:.2f}) < {seuil} : objectif atteint")
    else:
        print(f"\nMAE ({metrics['mae']:.2f}) >= {seuil} : objectif non atteint")

    plot_predictions(
        y_test, metrics["y_pred"], FIGURES_DIR / "01-predictions-vs-reel.png"
    )


if __name__ == "__main__":
    main()
