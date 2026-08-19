# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///
"""Analyse démographique du jeu de données Census Income (UCI).

Répond à 9 questions sur des données de recensement adulte en utilisant Pandas :
répartition par race, âge moyen des hommes, pourcentage de diplômés, etc.
"""

import pathlib
import sys
import urllib.request

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
DATA_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/"
    "boilerplate-demographic-data-analyzer/main/adult.data.csv"
)

# Colonnes attendues, vérifiées au chargement : le fichier distribué par
# freeCodeCamp porte déjà sa ligne d'en-tête, mais l'ordre et les noms
# conditionnent toutes les questions de l'énoncé.
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "salary",
]


def download_data(csv_path: pathlib.Path) -> None:
    """Télécharge adult.data.csv depuis freeCodeCamp si absent."""
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
    csv_path = DATA_DIR / "adult.data.csv"
    if not csv_path.exists():
        download_data(csv_path)
    # Le CSV porte sa propre ligne d'en-tête : forcer `header=None` la ferait
    # passer pour une observation, ce qui ajoute une ligne fantôme et bascule
    # les colonnes numériques en texte (« Cannot perform reduction 'mean' with
    # string dtype » au premier calcul).
    df = pd.read_csv(csv_path, skipinitialspace=True)

    if list(df.columns) != COLUMN_NAMES:
        raise ValueError(
            f"Colonnes inattendues dans {csv_path.name}.\n"
            f"  attendues : {COLUMN_NAMES}\n"
            f"  trouvées  : {list(df.columns)}"
        )
    return df


def _afficher(resultats: dict) -> None:
    """Affiche les résultats, séparé du calcul pour respecter `print_data`."""
    print("Nombre de personnes par race :")
    print(resultats["race_count"])
    print(f"\nÂge moyen des hommes : {resultats['average_age_men']}")
    print(f"% Bachelor's : {resultats['percentage_bachelors']}%")
    print(f"% éducation supérieure >50K : {resultats['higher_education_rich']}%")
    print(f"% sans éducation supérieure >50K : {resultats['lower_education_rich']}%")
    print(f"Heures min/semaine : {resultats['min_work_hours']}")
    print(f"% min heures >50K : {resultats['rich_percentage']}%")
    print(
        f"Pays + riche : {resultats['highest_earning_country']} "
        f"({resultats['highest_earning_country_percentage']}%)"
    )
    print(f"Occupation la + populaire >50K en Inde : {resultats['top_IN_occupation']}")


def calculate_demographic_data(print_data: bool = True) -> dict:
    """Calcule les 9 statistiques demandées par freeCodeCamp.

    La signature avec `print_data` est imposée : la cellule de correction
    appelle `calculate_demographic_data(print_data=False)`, et les clés du
    dictionnaire renvoyé sont lues telles quelles par les assertions.
    """
    df = load_data()

    # 1. Nombre de personnes par race
    race_count = df["race"].value_counts()

    # 2. Âge moyen des hommes
    average_age_men = df.loc[df["sex"] == "Male", "age"].mean()

    # 3. Pourcentage de personnes avec un Bachelor's degree
    percentage_bachelors = (df["education"] == "Bachelors").sum() / len(df) * 100

    # 4. % avec éducation avancée gagnant >50K
    advanced_education_mask = df["education"].isin(
        ["Bachelors", "Masters", "Doctorate"]
    )
    higher_education_rich = (
        df.loc[advanced_education_mask, "salary"] == ">50K"
    ).mean() * 100

    # 5. % sans éducation avancée gagnant >50K
    lower_education_rich = (
        df.loc[~advanced_education_mask, "salary"] == ">50K"
    ).mean() * 100

    # 6. Nombre minimum d'heures travaillées par semaine
    min_work_hours = df["hours-per-week"].min()

    # 7. % des personnes travaillant le minimum d'heures gagnant >50K
    min_workers = df[df["hours-per-week"] == min_work_hours]
    rich_percentage = (min_workers["salary"] == ">50K").mean() * 100

    # 8. Pays avec le plus haut pourcentage gagnant >50K
    rich_by_country = (
        df[df["salary"] == ">50K"].groupby("native-country").size()
        / df.groupby("native-country").size()
        * 100
    )
    highest_earning_country = rich_by_country.idxmax()
    highest_earning_country_percentage = rich_by_country.max()

    # 9. Occupation la plus populaire pour >50K en Inde
    india_rich = df[(df["native-country"] == "India") & (df["salary"] == ">50K")]
    top_IN_occupation = india_rich["occupation"].value_counts().idxmax()

    # Arrondi à une décimale : `assertAlmostEqual` compare à 7 décimales près,
    # et les valeurs attendues du corrigé (39.4, 16.4, 46.5…) sont arrondies.
    # Sans cet arrondi, 39.43354749885268 ne vaut pas 39.4.
    resultats = {
        "race_count": race_count,
        "average_age_men": average_age_men,
        "percentage_bachelors": percentage_bachelors,
        "higher_education_rich": higher_education_rich,
        "lower_education_rich": lower_education_rich,
        "min_work_hours": min_work_hours,
        "rich_percentage": rich_percentage,
        "highest_earning_country": highest_earning_country,
        "highest_earning_country_percentage": highest_earning_country_percentage,
        "top_IN_occupation": top_IN_occupation,
    }
    for _cle, _valeur in resultats.items():
        if isinstance(_valeur, float):
            resultats[_cle] = round(_valeur, 1)

    if print_data:
        _afficher(resultats)
    return resultats


def main() -> None:
    """Point d'entrée : calcule et affiche."""
    calculate_demographic_data(print_data=True)


if __name__ == "__main__":
    main()
