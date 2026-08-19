# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///
"""Tests unitaires pour le projet 02 : Demographic Data Analyzer."""

import pathlib
import re
import tempfile
import unittest
from unittest import mock

import pandas as pd
from demographic_data_analyzer import (
    calculate_demographic_data,
    download_data,
    load_data,
)


class TestDonnees(unittest.TestCase):
    """Vérifie la structure du jeu de données."""

    def setUp(self):
        self.df = load_data()

    def test_nombre_lignes(self):
        self.assertEqual(len(self.df), 32561)

    def test_colonnes_presentes(self):
        attendues = {
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
        }
        self.assertEqual(set(self.df.columns), attendues)

    def test_aucune_valeur_manquante(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)


class TestRaceCount(unittest.TestCase):
    """Vérifie le comptage par race."""

    def setUp(self):
        self.results = calculate_demographic_data()

    def test_est_serie(self):
        self.assertIsInstance(self.results["race_count"], pd.Series)

    def test_totale_egale_nombre_lignes(self):
        self.assertEqual(self.results["race_count"].sum(), 32561)


class TestAgeMoyen(unittest.TestCase):
    """Vérifie l'âge moyen des hommes."""

    def setUp(self):
        self.results = calculate_demographic_data()

    def test_entre_30_et_40(self):
        self.assertGreater(self.results["average_age_men"], 30)
        self.assertLess(self.results["average_age_men"], 40)


class TestPourcentages(unittest.TestCase):
    """Vérifie les pourcentages calculés."""

    def setUp(self):
        self.results = calculate_demographic_data()

    def test_bachelor_entre_10_et_20(self):
        self.assertGreater(self.results["percentage_bachelors"], 10)
        self.assertLess(self.results["percentage_bachelors"], 20)

    def test_higher_education_rich_entre_20_et_50(self):
        self.assertGreater(self.results["higher_education_rich"], 20)
        self.assertLess(self.results["higher_education_rich"], 50)

    def test_lower_education_rich_entre_0_et_20(self):
        self.assertGreater(self.results["lower_education_rich"], 0)
        self.assertLess(self.results["lower_education_rich"], 20)

    def test_min_work_hours_egale_1(self):
        self.assertEqual(self.results["min_work_hours"], 1)

    def test_rich_percentage_proche_10(self):
        self.assertGreater(self.results["rich_percentage"], 0)
        self.assertLess(self.results["rich_percentage"], 100)


class TestPays(unittest.TestCase):
    """Vérifie le pays avec le plus haut pourcentage >50K."""

    def setUp(self):
        self.results = calculate_demographic_data()

    def test_est_iran(self):
        self.assertEqual(self.results["highest_earning_country"], "Iran")

    def test_pourcentage_entre_30_et_50(self):
        self.assertGreater(self.results["highest_earning_country_percentage"], 30)
        self.assertLess(self.results["highest_earning_country_percentage"], 50)


class TestInde(unittest.TestCase):
    """Vérifie l'occupation la plus populaire >50K en Inde."""

    def setUp(self):
        self.results = calculate_demographic_data()

    est_occupation_valide = [
        "Tech-support",
        "Craft-repair",
        "Exec-managerial",
        "Prof-specialty",
        "Sales",
    ]

    def test_est_une_occupation_valide(self):
        self.assertIn(
            self.results["top_IN_occupation"],
            [
                "Tech-support",
                "Craft-repair",
                "Exec-managerial",
                "Prof-specialty",
                "Sales",
            ],
        )


class TestConformiteCorrigeOfficiel(unittest.TestCase):
    """Les 10 assertions de `test_module.py`, telles que le correcteur les lance.

    Les noms de clés font partie du contrat : le correcteur lit
    `data['average_age_men']`, pas `average_age_males`. Un renommage « plus
    clair » ferait échouer la soumission sans que le calcul soit faux.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = calculate_demographic_data(print_data=False)

    def test_cles_attendues_par_le_correcteur(self):
        attendues = {
            "race_count",
            "average_age_men",
            "percentage_bachelors",
            "higher_education_rich",
            "lower_education_rich",
            "min_work_hours",
            "rich_percentage",
            "highest_earning_country",
            "highest_earning_country_percentage",
            "top_IN_occupation",
        }
        self.assertTrue(attendues.issubset(set(self.data)))

    def test_valeurs_du_corrige(self):
        self.assertCountEqual(
            self.data["race_count"].tolist(), [27816, 3124, 1039, 311, 271]
        )
        self.assertAlmostEqual(self.data["average_age_men"], 39.4)
        self.assertAlmostEqual(self.data["percentage_bachelors"], 16.4)
        self.assertAlmostEqual(self.data["higher_education_rich"], 46.5)
        self.assertAlmostEqual(self.data["lower_education_rich"], 17.4)
        self.assertAlmostEqual(self.data["min_work_hours"], 1)
        self.assertAlmostEqual(self.data["rich_percentage"], 10)
        self.assertEqual(self.data["highest_earning_country"], "Iran")
        self.assertAlmostEqual(self.data["highest_earning_country_percentage"], 41.9)
        self.assertEqual(self.data["top_IN_occupation"], "Prof-specialty")

    def test_signature_accepte_print_data(self):
        """Le correcteur appelle avec `print_data=False` : la signature l'exige."""
        import inspect

        params = inspect.signature(calculate_demographic_data).parameters
        self.assertIn("print_data", params)


class TestChargement(unittest.TestCase):
    """Le fichier porte sa propre ligne d'en-tête."""

    def test_pas_de_ligne_fantome(self):
        """32561 lignes, pas 32562 : forcer header=None ajouterait les titres."""
        df = load_data()
        self.assertEqual(len(df), 32561)

    def test_colonnes_numeriques_bien_typees(self):
        """Une ligne d'en-tête prise pour une donnée basculerait age en texte."""
        df = load_data()
        for col in ("age", "fnlwgt", "education-num", "hours-per-week"):
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[col]), f"{col} n'est pas numérique"
            )


class TestChargementEchec(unittest.TestCase):
    """Le chemin d'erreur du téléchargement, seul code non couvert autrement."""

    def test_telechargement_impossible_arrete_le_programme(self):
        """Une panne réseau doit arrêter net.

        Sans `sys.exit(1)`, `load_data()` enchaîne sur la lecture d'un fichier
        absent et l'utilisateur reçoit un `FileNotFoundError` qui ne dit rien de
        la cause réelle.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cible = pathlib.Path(tmp) / "adult.data.csv"
            with mock.patch(
                "demographic_data_analyzer.urllib.request.urlopen",
                side_effect=OSError("réseau"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    download_data(cible)
        self.assertEqual(ctx.exception.code, 1)


class TestCoherenceDuReadme(unittest.TestCase):
    """Le README annonce un nombre de tests : il doit rester exact.

    Ce compteur est une donnée dupliquée entre le code et la documentation, sans
    rien pour les relier : il a déjà dérivé deux fois après un ajout de tests.
    Ce test échoue au prochain écart, ce qui est précisément son but.
    """

    def test_le_nombre_annonce_est_le_nombre_reel(self):
        import unittest as _unittest

        import test_units

        reel = _unittest.defaultTestLoader.loadTestsFromModule(
            test_units
        ).countTestCases()

        readme = (pathlib.Path(__file__).resolve().parent / "README.md").read_text(
            encoding="utf-8"
        )
        annonces = re.findall(r"(\d+) tests unitaires", readme)
        self.assertEqual(
            len(annonces), 1, "le README doit annoncer le nombre de tests une fois"
        )
        self.assertEqual(
            int(annonces[0]),
            reel,
            f"le README annonce {annonces[0]} tests, il y en a {reel} : "
            "mettre à jour la section « État ».",
        )


if __name__ == "__main__":
    unittest.main()
