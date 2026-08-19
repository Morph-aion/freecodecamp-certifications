# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///
"""Tests unitaires du projet Book Recommendation Engine (KNN).

La suite s'appuie sur `recommender.py`, le module de logique du projet : elle
lance le pipeline complet une seule fois (chargement, filtrage, matrice,
entraînement) et vérifie que la cellule de test officielle freeCodeCamp passe.

Deux catégories de tests ne demandent aucune donnée ni aucun modèle : le
filtrage sur un jeu synthétique, et la cohérence des références de la doc.

Lancement :
    uv run test_units.py
"""

import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from recommender import (
    N_RECOMMENDATIONS,
    REFERENCE_BOOK,
    BookRecommender,
    build_matrix,
    filter_participants,
    load_data,
    make_get_recommends,
)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Les distances attendues par la cellule de test officielle. Le correcteur
# compare avec une tolérance de 0.0001.
OFFICIAL_EXPECTED = [
    ("Catch 22", 0.793983519077301),
    ("The Witching Hour (Lives of the Mayfair Witches)", 0.7448656558990479),
    ("Interview with the Vampire", 0.7345068454742432),
    ("The Tale of the Body Thief (Vampire Chronicles (Paperback))", 0.5376338362693787),
    ("The Vampire Lestat (Vampire Chronicles, Book II)", 0.5178412199020386),
]


class TestModele(unittest.TestCase):
    """Le pipeline complet, mesuré sur le jeu de données réel.

    Entraîné une seule fois pour toute la classe : le pipeline (charge, filtre,
    pivote, entraîne) prend quelques secondes, ce n'est pas une unité de test en
    soi mais la matière de toutes les assertions qui suivent.
    """

    @classmethod
    def setUpClass(cls):
        if not RAW_DIR.is_dir():
            raise unittest.SkipTest("jeu de données absent, le télécharger d'abord")
        cls.books, cls.ratings = load_data()
        cls.filtered = filter_participants(cls.ratings)
        cls.matrix = build_matrix(cls.filtered, cls.books)
        cls.model = BookRecommender().fit(cls.matrix)

    def setUp(self):
        # Attribut d'instance, pas de classe : une fonction posée sur la classe
        # serait liée à `self` (méthode), et tout appel prendrait un argument de
        # trop. Sur l'instance, la fonction est un simple attribut.
        self.get_recommends = make_get_recommends(self.model)

    def test_la_cellule_de_test_officielle_passe(self):
        """Le critère d'acceptation de l'énoncé, tel quel."""
        import_books = self.get_recommends(REFERENCE_BOOK)
        test_books = import_books[1]

        self.assertEqual(len(test_books), 5)
        # `strict=True` : sans lui, `zip` tronquerait silencieusement sur la plus
        # courte des deux séquences, et une recommandation manquante passerait
        # pour un succès.
        for (titre_obtenu, distance_obtenue), (titre_attendu, distance_attendue) in zip(
            test_books, OFFICIAL_EXPECTED, strict=True
        ):
            self.assertEqual(titre_obtenu, titre_attendu)
            self.assertLess(abs(distance_obtenue - distance_attendue), 0.0001)

    def test_la_premiere_valeur_de_retour_est_le_titre_passe(self):
        """Contrat : `[titre, [[titre, distance] × 5]]`."""
        resultat = self.get_recommends(REFERENCE_BOOK)
        self.assertEqual(resultat[0], REFERENCE_BOOK)
        self.assertIsInstance(resultat[1], list)
        self.assertEqual(len(resultat[1]), N_RECOMMENDATIONS)
        for paire in resultat[1]:
            self.assertEqual(len(paire), 2)

    def test_les_recommandations_sont_triees_du_plus_eloigne_au_plus_proche(self):
        """Le sens de la liste est le « faux piège » du projet.

        `kneighbors` renvoie les voisins par distance croissante. freeCodeCamp
        attend la liste dans l'ordre inverse : le premier élément du corrigé est
        « Catch 22 », la recommandation *la plus lointaine* (0.7939), pas la plus
        proche. Ne pas inverser ferait échouer la cellule de test dès la première
        assertion.
        """
        resultat = self.get_recommends(REFERENCE_BOOK)
        distances = [paire[1] for paire in resultat[1]]
        self.assertEqual(distances, sorted(distances, reverse=True))
        self.assertEqual(resultat[1][0][0], "Catch 22")

    def test_le_livre_passe_en_argument_n_est_pas_recommande(self):
        """Le livre lui-même est le voisin à distance nulle : on l'écarte."""
        titres = [paire[0] for paire in self.get_recommends(REFERENCE_BOOK)[1]]
        self.assertNotIn(REFERENCE_BOOK, titres)

    def test_les_distances_du_corrige_sont_dans_l_ordre_du_corrige(self):
        """Garde-fou : confirme que `OFFICIAL_EXPECTED` est bien décroissant.

        Si le corrigé n'était pas trié comme la fonction, le test de tri ci-dessus
        serait une coïncidence à vérifier.
        """
        distances = [d for _, d in OFFICIAL_EXPECTED]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_le_livre_inconnu_leve_une_erreur_claire(self):
        with self.assertRaises(KeyError):
            self.get_recommends("Un livre qui n'existe pas")

    def test_la_matrice_filtree_a_la_taille_attendue(self):
        """673 livres × 888 utilisateurs après filtrage (seuils 200/100).

        Ce chiffre verrouille le pipeline : si le filtrage, le rattachement par
        ISBN ou le pivot dérivaient, la taille de la matrice bougerait.
        """
        self.assertEqual(self.matrix.shape, (673, 888))

    def test_les_chiffres_cites_par_la_documentation_sont_ceux_mesures(self):
        """Les chiffres de la doc valent une ancre, au même titre que ses lignes.

        `TestReferencesDeLaDocumentation` verrouille les numéros de ligne cités,
        mais rien ne verrouillait les mesures commentées dans le code et dans
        `notions-mobilisees.md`. C'est ainsi qu'un « 27 titres dupliqués » faux
        avait survécu. Chaque valeur affirmée est donc recalculée ici.
        """
        merged = pd.merge(
            left=self.filtered, right=self.books[["isbn", "title"]], on="isbn"
        )
        dedoublonne = merged.drop_duplicates(subset=["user", "title"])
        titres_multi_isbn = merged.groupby("title")["isbn"].nunique().gt(1).sum()
        non_nulles = int((self.matrix != 0).sum().sum())

        self.assertEqual(self.ratings["user"].nunique(), 105283, "utilisateurs bruts")
        self.assertEqual(titres_multi_isbn, 50, "titres portés par plusieurs ISBN")

        # La chaîne complète, maillon par maillon. Chaque nombre cité par la doc
        # se déduit du précédent : sans les intermédiaires, un décalage en amont
        # dériverait silencieusement jusqu'au bout.
        self.assertEqual(len(self.filtered), 49781, "notes après filtrage")
        self.assertEqual(len(merged), 49517, "notes rattachées à un titre")
        self.assertEqual(len(self.filtered) - len(merged), 264, "ISBN sans fiche livre")
        self.assertEqual(len(dedoublonne), 49136, "notes après dédoublonnage")
        self.assertEqual(len(merged) - len(dedoublonne), 381, "doublons (user, title)")
        self.assertEqual(non_nulles, 12425, "cellules non nulles")
        self.assertEqual(
            len(dedoublonne) - non_nulles, 36711, "notes à 0 dans la matrice"
        )

        # 37 141 et 36 711 ne disent pas la même chose : le premier compte les
        # zéros parmi les notes filtrées, le second ceux qui atteignent la
        # matrice, après les 264 et les 381 perdues en route. La doc cite les
        # deux, les deux sont donc ancrés.
        self.assertEqual(
            int((self.filtered["rating"] == 0).sum()), 37141, "zéros avant rattachement"
        )

        part_de_zeros = (self.filtered["rating"] == 0).mean() * 100
        self.assertAlmostEqual(part_de_zeros, 74.6, places=1)
        part_de_zeros_bruts = (self.ratings["rating"] == 0).mean() * 100
        self.assertAlmostEqual(part_de_zeros_bruts, 62.3, places=1)

    def test_le_tableau_d_invariance_de_la_doc(self):
        """Section 5 de notions-mobilisees.md : homothétie oui, translation non.

        Le tableau annonce 0,909955 / 0,909955 / 0,909956 pour deux lignes de la
        matrice réelle. L'homothétie est invariante, la translation non.

        Les comparaisons sont volontairement à tolérance plutôt qu'exactes. Une
        égalité au bit près tiendrait aujourd'hui mais dépendrait de l'ordre des
        sommations dans `cosine_distances` : une réécriture interne de numpy ou
        de scikit-learn ferait rougir le test sans qu'aucune propriété
        mathématique n'ait changé. Ce qu'on affirme, c'est que l'écart dû à
        l'homothétie est négligeable devant celui dû à la translation, et c'est
        cela qu'on mesure.
        """
        from sklearn.metrics.pairwise import cosine_distances

        ligne = self.matrix.loc["1984"].to_numpy(dtype="float64")
        temoin = self.matrix.loc["1st to Die: A Novel"].to_numpy(dtype="float64")
        base = float(cosine_distances([ligne], [temoin])[0, 0])
        homothetie = float(cosine_distances([ligne * 2], [temoin])[0, 0])
        translatee = ligne.copy()
        translatee[translatee != 0] += 2
        translation = float(cosine_distances([translatee], [temoin])[0, 0])

        self.assertAlmostEqual(base, 0.909955, places=5)
        self.assertAlmostEqual(homothetie, 0.909955, places=5)
        self.assertAlmostEqual(translation, 0.909956, places=5)

        # L'homothétie ne bouge pas au-delà du bruit flottant (~1e-16 relatif),
        # la translation déplace la distance d'un ordre de grandeur bien
        # supérieur. C'est le contraste qui porte l'affirmation de la doc.
        ecart_homothetie = abs(homothetie - base)
        ecart_translation = abs(translation - base)
        self.assertLess(ecart_homothetie, 1e-12, "l'homothétie doit être invariante")
        self.assertGreater(ecart_translation, 1e-9, "la translation doit déplacer")
        self.assertGreater(ecart_translation, ecart_homothetie * 1000)

    def test_la_correlation_de_popularite_de_la_doc(self):
        """Section 9 : la corrélation popularité × distance moyenne vaut -0,227.

        Un livre très noté est en moyenne un peu plus proche de tout le monde.
        """
        from sklearn.metrics.pairwise import cosine_distances

        X = self.matrix.to_numpy(dtype="float64")
        D = cosine_distances(X)
        np.fill_diagonal(D, np.nan)
        notes = (X != 0).sum(axis=1)
        distance_moyenne = np.nanmean(D, axis=1)
        correlation = float(np.corrcoef(notes, distance_moyenne)[0, 1])

        self.assertAlmostEqual(correlation, -0.227, places=3)


class TestFiltrage(unittest.TestCase):
    """Le filtrage sur un jeu synthétique, sans aucune donnée réelle."""

    def _jeu_synthetique(self):
        ratings = {
            "user": [1, 1, 1, 2, 2, 3, 3, 4],
            "isbn": ["A", "B", "C", "A", "B", "A", "B", "A"],
            "rating": [1, 2, 3, 4, 5, 6, 7, 8],
        }
        return pd.DataFrame(ratings)

    def test_un_utilisateur_sous_le_seuil_disparait(self):
        ratings = self._jeu_synthetique()
        # Seuils abaissés pour un jeu minuscule : au moins 2 notes par
        # utilisateur, au moins 3 notes par livre.
        filtre = filter_participants(ratings, min_user_ratings=2, min_book_ratings=3)
        # L'utilisateur 4 n'a qu'une note : exclu.
        self.assertNotIn(4, filtre["user"].values)
        # L'utilisateur 3 en a deux : conservé.
        self.assertIn(3, filtre["user"].values)

    def test_un_livre_sous_le_seuil_disparait(self):
        ratings = self._jeu_synthetique()
        filtre = filter_participants(ratings, min_user_ratings=1, min_book_ratings=3)
        # Le livre C n'a qu'une note : exclu.
        self.assertNotIn("C", filtre["isbn"].values)
        # Le livre A en a quatre : conservé.
        self.assertIn("A", filtre["isbn"].values)

    def test_les_seuils_par_defaut_appliquent_bien_200_et_100(self):
        """Les seuils de l'énoncé, vérifiés par ce que le filtrage fait.

        Inspecter la signature confirmerait seulement que quelqu'un a écrit 200
        et 100 quelque part : un corps de fonction vide passerait. On construit
        donc un jeu calibré juste autour des seuils : un utilisateur à 199 notes
        et un livre à 99 notes doivent tomber, leurs voisins à 200 et 100 rester.
        """
        # Les deux seuils s'appliquent ensemble : pour isoler l'effet de chacun,
        # il faut que tout le reste du jeu les franchisse largement. On part donc
        # d'un noyau dense (200 utilisateurs notant chacun les 200 mêmes livres,
        # ce qui met tout le monde au-dessus des deux seuils), puis on y ajoute
        # les deux cas limites à tester.
        lignes = [
            {"user": u, "isbn": f"B{i:04d}", "rating": 5}
            for u in range(200)
            for i in range(200)
        ]
        # Un utilisateur à 199 notes : juste sous le seuil de 200.
        lignes += [{"user": 900, "isbn": f"B{i:04d}", "rating": 5} for i in range(199)]
        # Un livre noté par 99 des utilisateurs du noyau : juste sous le seuil
        # de 100. Ces notes ne changent pas le sort de leurs auteurs, déjà à 200.
        lignes += [{"user": u, "isbn": "PRESQUE", "rating": 5} for u in range(99)]
        ratings = pd.DataFrame(lignes)

        filtre = filter_participants(ratings)

        self.assertIn(0, filtre["user"].values, "200 notes : au seuil, donc gardé")
        self.assertNotIn(900, filtre["user"].values, "199 notes : sous le seuil")
        self.assertIn("B0000", filtre["isbn"].values, "200 notes : au-dessus du seuil")
        self.assertNotIn("PRESQUE", filtre["isbn"].values, "99 notes : sous le seuil")


class TestJeuDeDonnees(unittest.TestCase):
    """Structure attendue sur le disque, si les données sont présentes."""

    def setUp(self):
        if not RAW_DIR.is_dir():
            self.skipTest("jeu de données absent, le télécharger d'abord")

    def test_les_trois_fichiers_du_jeu_existent(self):
        for nom in ("BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv"):
            with self.subTest(fichier=nom):
                self.assertTrue((RAW_DIR / nom).is_file())

    def test_les_donnees_ont_la_taille_annoncee(self):
        books, ratings = load_data()
        self.assertEqual(len(books), 271379)
        self.assertEqual(len(ratings), 1149780)


class TestChargementDesDonnees(unittest.TestCase):
    """Le docstring de `load_data` décrit un comportement de pandas, pas un contrat.

    La règle sur `names` / `usecols` a été mesurée, pas déduite d'une garantie
    de la bibliothèque : elle peut changer d'une version à l'autre, et `uv`
    résout la version au moment de l'exécution. Figer un numéro dans le
    docstring l'aurait périmé en silence. Ces tests le revérifient à chaque
    exécution, sur le vrai fichier.
    """

    def setUp(self):
        if not RAW_DIR.is_dir():
            self.skipTest("jeu de données absent, le télécharger d'abord")
        self.chemin = RAW_DIR / "BX-Books.csv"
        self.commun = dict(encoding="ISO-8859-1", sep=";", header=0)

    def test_la_regle_names_usecols_decrite_par_le_docstring_tient_toujours(self):
        """Trois combinaisons passent, une échoue : c'est ce que dit `load_data`."""
        deux_deux = dict(names=["isbn", "title"], usecols=["isbn", "title"])
        huit_deux = dict(
            names=[
                "isbn",
                "title",
                "author",
                "year",
                "publisher",
                "image_s",
                "image_m",
                "image_l",
            ],
            usecols=["isbn", "title"],
        )
        for etiquette, kwargs in [("2/2", deux_deux), ("8/2", huit_deux)]:
            with self.subTest(combinaison=etiquette):
                lu = pd.read_csv(self.chemin, **self.commun, **kwargs, dtype="str")
                self.assertEqual(list(lu.columns), ["isbn", "title"])
                self.assertEqual(len(lu), 271379)

        # Le panachage : `names` plus long que `usecols` mais plus court que le
        # fichier. C'est le cas qui lève, et c'est lui qui justifie le choix
        # retenu dans `load_data`. S'il cessait d'échouer, le docstring
        # deviendrait faux et ce test le dirait.
        with self.assertRaises(
            ValueError, msg="pandas n'échoue plus sur 3/2 : mettre à jour load_data"
        ):
            pd.read_csv(
                self.chemin,
                **self.commun,
                names=["isbn", "title", "author"],
                usecols=["isbn", "title"],
                dtype="str",
            )

    def test_les_colonnes_chargees_sont_bien_celles_du_fichier(self):
        """Un mauvais mapping ne lèverait rien : il décalerait silencieusement.

        `names` renomme par position. Si l'ordre des colonnes du fichier
        changeait, `title` pourrait recevoir les ISBN sans qu'aucune exception
        ne soit levée. On compare donc à une relecture brute, par position.
        """
        books, _ = load_data()
        brut = pd.read_csv(self.chemin, **self.commun, dtype="str", nrows=5)

        self.assertEqual(list(brut.columns)[:2], ["ISBN", "Book-Title"])
        for i in range(3):
            with self.subTest(ligne=i):
                self.assertEqual(books.iloc[i]["isbn"], brut.iloc[i, 0])
                self.assertEqual(books.iloc[i]["title"], brut.iloc[i, 1])

    def test_l_encodage_restitue_les_titres_non_ascii(self):
        """ISO-8859-1 mal choisi ne lèverait pas : il corromprait les accents.

        Deux titres identiques décodés différemment deviendraient deux livres
        distincts, et fausseraient le regroupement par titre de `build_matrix`.
        """
        books, _ = load_data()
        non_ascii = books["title"].str.contains(r"[^\x00-\x7F]", na=False, regex=True)
        self.assertEqual(int(non_ascii.sum()), 6635)


class TestReferencesDeLaDocumentation(unittest.TestCase):
    """`notions-mobilisees.md` cite des lignes précises de `recommender.py`.

    Ces références se périment silencieusement dès qu'on ajoute du code au-dessus
    d'elles. Ce test attache chaque numéro cité à ce qu'il est censé désigner.
    """

    ANCRES = {
        48: "def load_data",
        93: "def filter_participants",
        105: "user_counts = ratings",
        111: "].copy()",
        114: "def build_matrix",
        141: "class BookRecommender",
        154: "def __init__",
        177: "def _require_fitted",
        182: "def fit",
        187: "def recommend",
        223: "def make_get_recommends",
    }

    def setUp(self):
        with open(BASE_DIR / "recommender.py", encoding="utf-8") as f:
            self.source = f.read().splitlines()

    def test_chaque_ligne_citee_contient_ce_qu_elle_annonce(self):
        for numero, motif in sorted(self.ANCRES.items()):
            with self.subTest(ligne=numero, attendu=motif):
                self.assertLessEqual(numero, len(self.source))
                self.assertIn(
                    motif,
                    self.source[numero - 1],
                    f"la ligne {numero} a bougé, mettre à jour notions-mobilisees.md",
                )

    def test_toutes_les_references_du_document_sont_couvertes(self):
        chemin = BASE_DIR / "docs" / "notions-mobilisees.md"
        if not chemin.is_file():
            self.skipTest("document de notions absent")
        with open(chemin, encoding="utf-8") as f:
            document = f.read()

        citees = set()
        for debut, fin in re.findall(r"recommender\.py#L(\d+)(?:-L(\d+))?", document):
            citees.add(int(debut))
            if fin:
                citees.add(int(fin))

        non_couvertes = citees - set(self.ANCRES)
        self.assertEqual(
            non_couvertes,
            set(),
            f"lignes citées sans ancre de vérification : {sorted(non_couvertes)}",
        )

    def test_les_liens_du_document_pointent_vers_un_fichier_existant(self):
        """Vérifier les numéros ne sert à rien si le chemin ne résout pas.

        `notions-mobilisees.md` vit dans `docs/`, donc un lien vers
        `recommender.py` doit remonter d'un cran (`../recommender.py`). L'erreur
        tient à deux caractères et reste invisible sans ce test.
        """
        chemin = BASE_DIR / "docs" / "notions-mobilisees.md"
        if not chemin.is_file():
            self.skipTest("document de notions absent")
        document = chemin.read_text(encoding="utf-8")

        for cible in re.findall(r"\]\(([^)]+\.py)(?:#L\d+(?:-L\d+)?)?\)", document):
            with self.subTest(lien=cible):
                self.assertTrue(
                    (chemin.parent / cible).resolve().is_file(),
                    f"lien cassé depuis docs/ : {cible}",
                )

    def test_le_nombre_de_tests_annonce_par_le_readme_est_le_bon(self):
        """Le README annonce un compte de tests, qui se périme à chaque ajout.

        C'est arrivé deux fois pendant l'écriture de ce projet : ajouter une
        classe de tests laisse le chiffre du README en arrière, et rien ne le
        signale. Le compte est donc dérivé de la suite elle-même.
        """
        chemin = BASE_DIR / "README.md"
        if not chemin.is_file():
            self.skipTest("README absent")

        suite = unittest.defaultTestLoader.loadTestsFromName("test_units")
        reel = suite.countTestCases()

        annonce = re.search(r"(\d+) tests verts", chemin.read_text(encoding="utf-8"))
        self.assertIsNotNone(annonce, "le README n'annonce plus de compte de tests")
        self.assertEqual(
            int(annonce.group(1)),
            reel,
            f"le README annonce {annonce.group(1)} tests, la suite en compte {reel}",
        )


class TestCoherenceDuNotebook(unittest.TestCase):
    """Le notebook contient du texte imposé qui peut diverger du module.

    La cellule de test officielle est recopiée telle quelle depuis l'énoncé :
    titre en dur, distances en dur. C'est voulu, c'est le livrable soumis au
    correcteur. Mais rien ne relie ce texte figé à `REFERENCE_BOOK` ni à
    `OFFICIAL_EXPECTED` : changer la constante laisserait le notebook sur
    l'ancien titre sans qu'aucun test ne bronche. Ces vérifications ferment
    l'écart sans toucher à la cellule.
    """

    def setUp(self):
        self.source = (BASE_DIR / "notebook.py").read_text(encoding="utf-8")

    def test_le_titre_en_dur_du_notebook_est_celui_de_la_constante(self):
        """Le titre doit être présent, sa mise en page importe peu.

        Chercher `get_recommends("<titre>")` d'un seul tenant était trop
        rigide : `ruff format` coupe l'appel sur trois lignes quand il dépasse
        88 colonnes, et le test rougissait alors que rien de substantiel
        n'avait changé. On cherche donc le littéral seul.
        """
        self.assertIn(
            f'"{REFERENCE_BOOK}"',
            self.source,
            "REFERENCE_BOOK a changé, la cellule officielle du notebook non",
        )
        self.assertIn(
            "get_recommends(",
            self.source,
            "la cellule officielle n'appelle plus get_recommends",
        )

    def test_les_distances_en_dur_du_notebook_sont_celles_du_corrige(self):
        for _, distance in OFFICIAL_EXPECTED:
            with self.subTest(distance=distance):
                self.assertIn(
                    str(distance),
                    self.source,
                    "OFFICIAL_EXPECTED a changé, la cellule du notebook non",
                )

    def test_le_notebook_parse_sous_la_version_qu_il_declare(self):
        """La PEP 723 doit annoncer une version qui sait lire le fichier.

        Le piège a été rencontré : une f-string réutilisant des guillemets
        doubles dans son expression (`f"{ratings["rating"]}"`) ne parse qu'à
        partir de 3.12, par la PEP 701. Le fichier annonçait `>=3.11`, donc
        `uv` pouvait résoudre vers un interpréteur où l'import échouait avec une
        `SyntaxError`, sans que rien ne l'ait signalé auparavant.

        Le notebook est revenu aux guillemets simples, ce qui l'aligne sur
        `>=3.11` comme les projets 01 et 02. Ce test empêche la régression :
        il compile réellement la source sous la version déclarée.

        Deux vérifications, et non une. La première suit la version annoncée,
        quelle qu'elle soit. La seconde impose 3.11 en dur : sans elle, relever
        `requires-python` à 3.12 ferait disparaître la garde au moment précis où
        la syntaxe se met à diverger de celle des projets 01 et 02. Si un jour
        le notebook a besoin de 3.12, c'est cette seconde assertion qu'il faudra
        retirer sciemment, plutôt qu'une garde qui s'éteint toute seule.
        """
        import ast

        version = re.search(r'requires-python = ">=(\d+)\.(\d+)"', self.source)
        self.assertIsNotNone(version, "en-tête PEP 723 introuvable")
        declaree = (int(version.group(1)), int(version.group(2)))

        # `feature_version` fait parser l'AST avec la grammaire de la version
        # visée, sans avoir besoin d'un interpréteur de cette version.
        for cible, motif in [
            (declaree, "la version déclarée par l'en-tête PEP 723"),
            ((3, 11), "le plancher commun aux projets 01 et 02"),
        ]:
            with self.subTest(python=f"{cible[0]}.{cible[1]}"):
                try:
                    ast.parse(self.source, feature_version=cible)
                except SyntaxError as erreur:
                    self.fail(
                        f"notebook.py ne parse pas en {cible[0]}.{cible[1]} "
                        f"({motif}) : {erreur.msg} ligne {erreur.lineno}. "
                        f"Retirer la syntaxe trop récente, ou relever "
                        f"requires-python et assumer l'écart avec 01 et 02."
                    )


if __name__ == "__main__":
    unittest.main()
