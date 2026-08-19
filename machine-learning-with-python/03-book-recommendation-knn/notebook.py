# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas",
#     "scikit-learn",
#     "matplotlib",
# ]
# ///

# Notebook du projet 03 : Book Recommendation Engine using KNN.
# Aucune logique ici : tout vient de recommender.py (convention du projet 01). Ce notebook
# reprend la structure imposée par l'énoncé freeCodeCamp (les premières cellules
# importent bibliothèques et données, la dernière sert de test), puis ajoute
# l'exploration visuelle.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from recommender import (
        REFERENCE_BOOK,
        BookRecommender,
        build_matrix,
        filter_participants,
        load_data,
        make_get_recommends,
    )

    return (
        REFERENCE_BOOK,
        BookRecommender,
        build_matrix,
        filter_participants,
        load_data,
        make_get_recommends,
        mo,
        plt,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Book Recommendation Engine using KNN

        Le jeu de données **Book-Crossings** contient 1,1 million de notes sur
        270 000 livres. L'objectif : pour un livre donné, trouver les 5 livres les
        plus proches, au sens de la **distance cosinus** entre leurs profils de
        notes.

        Une chose à savoir avant de lire quoi que ce soit : l'échelle annoncée va
        de 1 à 10, mais **62,3 % des notes valent en réalité 0** : ce sont des
        interactions enregistrées sans jugement explicite. La section 1 le
        recalcule sur les données chargées.

        Ce notebook est un squelette prêt à travailler : données chargées, filtrage
        appliqué, modèle entraîné, cellule de test officielle en dernière position.
        Chaque cellule se réexécute quand ce dont elle dépend change.
        """
    )
    return


@app.cell
def _(load_data, mo):
    books, ratings = load_data()
    zeros_bruts = f"{(ratings['rating'] == 0).mean() * 100:.1f}".replace(".", ",")
    mo.md(
        f"""
        ## 1. Les données

        **{len(ratings):,}** notes pour **{len(books):,}** livres, données par
        **{ratings["user"].nunique():,}** utilisateurs. L'énoncé arrondit ce
        dernier chiffre à 90 000.

        Dont **{zeros_bruts} %** de notes à `0`,
        valeur pourtant absente de l'échelle 1-10 annoncée : ce sont les
        interactions sans note explicite.
        """
    )
    return books, ratings


@app.cell
def _(mo, books, ratings):
    mo.vstack(
        [
            mo.md("Les notes brutes, colonnes `user`, `isbn`, `rating` :"),
            mo.ui.table(ratings.head()),
            mo.md("Les livres, avec leur titre :"),
            mo.ui.table(books.head()),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Le filtrage

        En regardant la distribution, on voit que la plupart des livres sont rarement
        notés : un livre noté deux fois, ou un utilisateur qui n'a noté que trois
        livres, n'offre presque aucune matière à la distance cosinus. L'énoncé impose
        donc de retirer :
        """
    )
    return


@app.cell
def _(filter_participants, ratings):
    filtered = filter_participants(ratings)
    return (filtered,)


@app.cell
def _(filtered, mo, ratings):
    book_counts = ratings["isbn"].value_counts()
    zeros_filtres = f"{(filtered['rating'] == 0).mean() * 100:.1f}".replace(".", ",")

    mo.md(
        f"""
        Après filtrage (**utilisateurs ayant au moins 200 notes** et **livres ayant
        au moins 100 notes**), il reste **{len(filtered):,}** notes
        (contre {len(ratings):,}), données par **{filtered["user"].nunique():,}**
        utilisateurs sur **{filtered["isbn"].nunique():,}** livres.

        À noter : la part de notes à `0` **monte** à
        **{zeros_filtres} %**. Le seuil « au moins
        100 notes » sélectionne donc surtout des livres avec lesquels beaucoup de
        gens ont interagi, pas des livres beaucoup jugés.
        """
    )
    return (book_counts,)


@app.cell
def _(book_counts, plt):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(book_counts.values, bins=40, log=True)
    ax.axvline(100, color="crimson", linestyle="--", label="seuil de 100 notes")
    ax.set_xlabel("Nombre de notes par livre")
    ax.set_ylabel("Nombre de livres (échelle log)")
    ax.set_title("La plupart des livres sont rarement notés")
    ax.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. La matrice et le modèle

        On pivote les notes en une matrice **livres × utilisateurs** : chaque ligne
        est le profil de notes d'un livre, chaque colonne celle d'un utilisateur. Les
        notes manquantes deviennent 0. Deux livres sont similaires si leurs profils
        pointent dans la même direction : c'est exactement ce que mesure la distance
        cosinus.
        """
    )
    return


@app.cell
def _(books, build_matrix, filtered):
    matrix = build_matrix(filtered, books)
    return (matrix,)


@app.cell
def _(filtered, matrix, mo):
    remplies = matrix != 0
    total_cellules = matrix.shape[0] * matrix.shape[1]
    parcimonie = f"{remplies.mean().mean() * 100:.2f}".replace(".", ",")

    mo.md(
        f"""
        Matrice **{matrix.shape[0]}** livres × **{matrix.shape[1]}** utilisateurs, avec
        **{remplies.sum().sum():,}** cellules non nulles sur
        **{total_cellules:,}**, soit une parcimonie de
        **{parcimonie} %**.

        Le rapprochement qui compte : **{len(filtered):,}** notes ont été retenues
        par le filtrage, mais seules **{remplies.sum().sum():,}** cellules sont
        non nulles. L'écart ne se réduit pas aux notes à `0` : 264 notes se perdent
        au rattachement des ISBN et 381 doublons sont écartés avant le pivot, et
        le `fillna(0)` rend les notes à 0 restantes indiscernables des absences.
        """
    )
    return


@app.cell
def _(BookRecommender, make_get_recommends, matrix):
    model = BookRecommender().fit(matrix)
    get_recommends = make_get_recommends(model)
    return get_recommends, model


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Le modèle

        `NearestNeighbors` (métrique cosinus) est entraîné sur la matrice. La
        fonction `get_recommends(titre)` renvoie `[titre, [[titre, distance] × 5]]`,
        les recommandations étant ordonnées **du plus éloigné au plus proche**.
        """
    )
    return


@app.cell
def _(REFERENCE_BOOK, get_recommends, mo):
    example = get_recommends(REFERENCE_BOOK)
    mo.ui.table(
        [
            {"recommandation": titre, "distance": round(distance, 4)}
            for titre, distance in example[1]
        ]
    )
    return (example,)


@app.cell
def _(mo):
    mo.md("## 5. La cellule de test officielle (dernière cellule)")
    return


@app.cell
def _(get_recommends):
    # Cellule reproduite telle quelle depuis l'énoncé freeCodeCamp : titre en
    # dur, noms de variables et assertions inclus. Ne pas la refactoriser pour
    # utiliser REFERENCE_BOOK, c'est le livrable que le correcteur attend. Le
    # test équivalent, lui, passe bien par la constante (`test_units.py`), et
    # `TestCoherenceDuNotebook` vérifie que les deux ne divergent pas.
    import_books = get_recommends(
        "The Queen of the Damned (Vampire Chronicles (Paperback))"
    )
    test_books = import_books[1]
    test_books_ratings = [book[1] for book in test_books]

    assert len(test_books) == 5
    assert test_books[0][0] == "Catch 22"
    assert abs(test_books[0][1] - 0.793983519077301) < 0.0001
    assert abs(test_books[1][1] - 0.7448656558990479) < 0.0001
    assert abs(test_books[2][1] - 0.7345068454742432) < 0.0001
    assert abs(test_books[3][1] - 0.5376338362693787) < 0.0001
    assert abs(test_books[4][1] - 0.5178412199020386) < 0.0001

    print("You passed the challenge!")
    return


if __name__ == "__main__":
    app.run()
