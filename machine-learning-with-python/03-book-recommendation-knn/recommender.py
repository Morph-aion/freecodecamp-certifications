"""Logique du projet 03 : Book Recommendation Engine using KNN.

Tout le raisonnement calculable vit ici ; le notebook (`notebook.py`) n'orchestre
et ne visualise que, conformément à la convention du projet 01. Les décisions qui font que le
projet se fait sereinement sont toutes portées par ce module :

1. les seuils de l'énoncé (utilisateurs d'au moins 200 notes, livres d'au moins
   100 notes), appliqués par `filter_participants` ;
2. le sens de la liste renvoyée, du plus éloigné au plus proche, l'ordre que
   freeCodeCamp attend dans la cellule de test, dans `BookRecommender.recommend` ;
3. le traitement des notes à 0, qui sont 3/4 du jeu retenu et non un cas
   marginal : voir `build_matrix` ;
4. la reproductibilité : données locales versionnées hors Git, matrice dense
   float64, et métrique cosinus explicite avec `algorithm="brute"`, le seul
   algorithme de `NearestNeighbors` qui accepte cette métrique, les arbres
   KD/Ball exigeant une vraie distance métrique.

Le contrat freeCodeCamp, `get_recommends(titre)`, est fabriqué par
`make_get_recommends`, appliqué sur un modèle entraîné : la fonction et le modèle
restent découplés, la fonction est testable sans état global.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Les données brutes ne sont jamais versionnées : elles sont reproductibles à
# l'identique depuis cette URL, qui doit rester documentée ici et pas seulement
# dans le README (convention de la certification). Décompresser l'archive dans
# `data/raw/` pour retrouver les trois fichiers `BX-*.csv`.
DATA_URL = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
BOOKS_FILENAME = "BX-Books.csv"
RATINGS_FILENAME = "BX-Book-Ratings.csv"

REFERENCE_BOOK = "The Queen of the Damned (Vampire Chronicles (Paperback))"

MIN_USER_RATINGS = 200
MIN_BOOK_RATINGS = 100
N_RECOMMENDATIONS = 5


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les notes et les livres du jeu de données Book-Crossings.

    Le jeu est livré tel quel par freeCodeCamp, séparateur `;`, encodage
    ISO-8859-1. Le fichier des livres compte huit colonnes (auteur, année,
    éditeur, trois URLs d'images...) dont deux seulement servent : le modèle ne
    lit aucun attribut d'un livre, seulement qui l'a noté. Les autres ne sont
    même pas chargées, pour ne pas monter 78 Mo en mémoire en vain.

    Sur la combinaison `names` / `usecols`, mesuré sur ce fichier plutôt que
    déduit : `names` de longueur 2 avec `usecols` de longueur 2 fonctionne, tout
    comme 8 et 2. C'est le panachage qui casse : `names` de longueur 3 avec
    `usecols` de longueur 2 lève « Number of passed names did not match number
    of header fields in the file ». La règle retenue est donc que `names` doit
    soit couvrir toutes les colonnes du fichier, soit avoir exactement la
    longueur de `usecols`. C'est cette seconde voie qu'on prend, la plus courte
    à lire.

    Cette règle décrit le comportement de la version de pandas que `uv` résout
    au moment où l'on exécute, pas un contrat stable de la bibliothèque : elle
    pourrait changer sans préavis. Plutôt que de figer un numéro de version qui
    se périmerait en silence, `test_units.py` la revérifie à chaque exécution
    (`TestChargementDesDonnees`), et signalera l'écart le jour où il apparaît.
    """
    books = pd.read_csv(
        data_dir / BOOKS_FILENAME,
        encoding="ISO-8859-1",
        sep=";",
        header=0,
        names=["isbn", "title"],
        usecols=["isbn", "title"],
        dtype={"isbn": "str", "title": "str"},
    )
    ratings = pd.read_csv(
        data_dir / RATINGS_FILENAME,
        encoding="ISO-8859-1",
        sep=";",
        header=0,
        names=["user", "isbn", "rating"],
        usecols=["user", "isbn", "rating"],
        dtype={"user": "int32", "isbn": "str", "rating": "float64"},
    )
    return books, ratings


def filter_participants(
    ratings: pd.DataFrame,
    min_user_ratings: int = MIN_USER_RATINGS,
    min_book_ratings: int = MIN_BOOK_RATINGS,
) -> pd.DataFrame:
    """Retire les participants trop peu actifs pour être statistiquement parlants.

    Un utilisateur qui a noté 3 livres, ou un livre noté 2 fois, ajoute du bruit
    sans matière : la distance n'aurait presque rien à mesurer. L'énoncé impose
    donc de retirer les utilisateurs à moins de 200 notes et les livres à moins
    de 100 notes.
    """
    user_counts = ratings["user"].value_counts()
    book_counts = ratings["isbn"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    rated_books = book_counts[book_counts >= min_book_ratings].index
    return ratings[
        ratings["user"].isin(active_users) & ratings["isbn"].isin(rated_books)
    ].copy()


def build_matrix(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice livres × utilisateurs, indexée par titre.

    Les notes sont rattachées aux livres par ISBN, puis réunies sur le titre : un
    même titre porté par plusieurs ISBN se retrouve dans une seule ligne (50
    titres sont dans ce cas dans la matrice filtrée). Les doublons
    utilisateur-titre sont écartés avant le pivot pour garantir l'unicité de
    chaque cellule, soit 381 lignes sur le jeu filtré.

    Le `fillna(0)` mérite une mise en garde. Il rend le 0 ambigu : une case à 0
    peut vouloir dire « cet utilisateur n'a pas noté ce livre » aussi bien que
    « il l'a noté 0 ». Or 74,6 % des notes retenues valent déjà 0 : ce sont les
    interactions sans note explicite du jeu Book-Crossings, pas des cas
    marginaux. Sur 49 781 notes filtrées, seules 12 425 cellules sont non nulles.

    On l'accepte quand même, pour deux raisons : c'est ce que fait le corrigé
    officiel, donc la seule façon de reproduire ses distances ; et 0 est l'élément
    neutre du produit scalaire, là où une note moyenne rapprocherait tous les
    livres par défaut. Conséquence à garder en tête pour interpréter les
    résultats : le modèle mesure surtout de la co-interaction, pas de la
    similarité de goût.
    """
    merged = pd.merge(left=ratings, right=books[["isbn", "title"]], on="isbn")
    merged = merged.drop_duplicates(subset=["user", "title"])
    return pd.pivot(merged, index="title", columns="user", values="rating").fillna(0)


class BookRecommender:
    """Modèle k-plus-proches-voisins sur la matrice livres × utilisateurs.

    La distance cosinus mesure l'angle entre deux lignes de la matrice : deux
    livres sont proches si leurs profils de notes pointent dans la même
    direction, quelle que soit leur norme. L'invariance est donc celle d'une
    homothétie (multiplier une ligne entière par un facteur ne change pas la
    distance) et non celle d'une translation : ajouter 2 à toutes les notes d'un
    profil la fait bouger. Un utilisateur « généreux » qui note tout haut relève
    plutôt du second cas, et il joue de surcroît sur les colonnes, pas sur les
    lignes comparées ici : le cosinus ne corrige pas ce biais-là.
    """

    def __init__(
        self,
        n_neighbors: int = N_RECOMMENDATIONS,
        metric: str = "cosine",
    ) -> None:
        # Deux compteurs voisins qu'il ne faut pas confondre : ce que l'appelant
        # veut recevoir, et ce que scikit-learn doit chercher. Le voisin le plus
        # proche d'un livre étant toujours lui-même, à distance nulle, le second
        # vaut le premier plus un. D'où un nom explicite côté nôtre, pour ne pas
        # doubler le `n_neighbors` de `self.nn` avec une autre sémantique.
        self.n_recommendations = n_neighbors
        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric=metric,
            algorithm="brute",
        )
        self._matrix: pd.DataFrame | None = None

    @property
    def titles(self) -> pd.Index:
        self._require_fitted()
        return self._matrix.index

    def _require_fitted(self) -> None:
        """Un seul message pour un seul état invalide, quel qu'en soit l'accès."""
        if self._matrix is None:
            raise RuntimeError("Modèle non entraîné : appeler fit() d'abord.")

    def fit(self, matrix: pd.DataFrame) -> BookRecommender:
        self._matrix = matrix
        self.nn.fit(matrix.to_numpy(dtype="float64"))
        return self

    def recommend(self, book: str, n: int | None = None) -> list[tuple[str, float]]:
        """Renvoie les n livres les plus proches de `book`.

        `n` vaut par défaut le `n_neighbors` donné au constructeur ; le passer
        ici ne sert qu'à interroger ponctuellement un autre nombre de voisins.

        Demander plus de voisins qu'à l'entraînement fonctionne, mais tient à
        `algorithm="brute"` : le balayage exhaustif calcule toutes les distances
        à chaque requête, le `n_neighbors` du constructeur n'étant qu'un défaut.
        Un arbre KD ou Ball, lui, exploite cette valeur à la construction et
        peut échouer. La contrainte est sans conséquence ici, le cosinus
        imposant déjà `brute`, mais le lien mérite d'être connu.

        `kneighbors` demande n+1 voisins : le premier est le livre lui-même, à
        distance nulle. La liste de sortie est volontairement ordonnée du plus
        éloigné au plus proche : c'est le sens que la cellule de test de
        freeCodeCamp attend. Le livre passé en argument n'y figure jamais.
        """
        self._require_fitted()
        if n is None:
            n = self.n_recommendations
        if book not in self._matrix.index:
            raise KeyError(
                f"« {book} » n'a pas assez de notes pour apparaître dans la "
                f"matrice filtrée ({len(self._matrix)} livres retenus)."
            )
        row = self._matrix.loc[[book]].to_numpy(dtype="float64")
        distances, indices = self.nn.kneighbors(row, n_neighbors=n + 1)
        recommendations = [
            (self.titles[indices[0, i]], float(distances[0, i]))
            for i in range(1, n + 1)
        ]
        recommendations.reverse()
        return recommendations


def make_get_recommends(
    model: BookRecommender,
    n: int | None = None,
) -> Callable[[str], list]:
    """Fabrique la fonction `get_recommends` attendue par la cellule de test.

    Contrat freeCodeCamp : `[titre, [[titre, distance] × 5]]`, où les
    recommandations sont ordonnées de la plus lointaine à la plus proche.

    `n` laissé à None suit le réglage du modèle, pour qu'un seul endroit décide
    du nombre de recommandations.
    """

    def get_recommends(book: str = ""):
        return [
            book,
            [[title, distance] for title, distance in model.recommend(book, n)],
        ]

    return get_recommends
