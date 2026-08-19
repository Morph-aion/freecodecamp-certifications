# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Tests unitaires pour le projet 05 : Sea Level Predictor."""

import pathlib
import re
import tempfile
import unittest
from unittest import mock

import matplotlib

# Backend non interactif : les tests tracent des figures sans écran.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import linregress  # noqa: E402
from sea_level_predictor import (  # noqa: E402
    download_data,
    draw_plot,
    load_data,
)


class TestDonnees(unittest.TestCase):
    """Vérifie la structure du jeu de données."""

    def setUp(self):
        self.df = load_data()

    def test_nombre_lignes(self):
        self.assertGreater(len(self.df), 100)

    def test_colonnes_presentes(self):
        self.assertIn("Year", self.df.columns)
        self.assertIn("CSIRO Adjusted Sea Level", self.df.columns)

    def test_valeurs_manquantes_localisees(self):
        """113 valeurs manquantes, toutes dans « NOAA Adjusted Sea Level ».

        La série NOAA ne commence qu'en 1993 : les années antérieures sont
        vides. La colonne utilisée par le projet, « CSIRO Adjusted Sea Level »,
        est complète sur les 134 années.
        """
        manquantes = self.df.isnull().sum()
        self.assertEqual(manquantes["CSIRO Adjusted Sea Level"], 0)
        self.assertEqual(manquantes["Year"], 0)
        self.assertEqual(manquantes["NOAA Adjusted Sea Level"], 113)

    def test_plage_dates(self):
        """Les données commencent avant 1900."""
        self.assertLess(self.df["Year"].min(), 1900)

    def test_niveau_est_un_ecart_croissant(self):
        """Le niveau est un écart signé, pas une hauteur positive."""
        # Le niveau est un écart à une référence, pas une hauteur absolue : il
        # est négatif avant 1884 (minimum -0,44 pouce en 1882). Ce qui est
        # vérifiable, c'est la tendance croissante sur le siècle.
        colonne = self.df["CSIRO Adjusted Sea Level"]
        self.assertLess(colonne.min(), 0)
        self.assertGreater(colonne.iloc[-1], colonne.iloc[0])


class TestRegression(unittest.TestCase):
    """Vérifie la régression linéaire."""

    def setUp(self):
        self.df = load_data()
        self.slope, self.intercept, self.r, *_ = linregress(
            self.df["Year"], self.df["CSIRO Adjusted Sea Level"]
        )

    def test_pente_positive(self):
        """Le niveau de la monte."""
        self.assertGreater(self.slope, 0)

    def test_r_fort(self):
        """Corrélation forte entre année et niveau de la mer."""
        self.assertGreater(self.r, 0.9)

    def test_r2_eleve(self):
        r2 = self.r**2
        self.assertGreater(r2, 0.8)


class TestDrawPlot(unittest.TestCase):
    """Vérifie le graphique."""

    def setUp(self):
        self.df = load_data()
        # draw_plot renvoie un Axes, pas une Figure : c'est le contrat imposé
        # par le correcteur, qui fait `ax = draw_plot()`.
        self.ax = draw_plot(self.df)
        self.fig = self.ax.figure

    def tearDown(self):
        plt.close(self.fig)

    def test_retourne_figure(self):
        self.assertIsInstance(self.fig, plt.Figure)

    def test_titre(self):
        ax = self.ax
        self.assertIn("Rise in Sea Level", ax.get_title())

    def test_label_x(self):
        ax = self.ax
        self.assertEqual(ax.get_xlabel(), "Year")

    def test_label_y(self):
        ax = self.ax
        self.assertEqual(ax.get_ylabel(), "Sea Level (inches)")

    def test_deux_lignes_tendance(self):
        """Deux lignes rouges et vertes + scatter."""
        ax = self.ax
        lignes = [ligne for ligne in ax.get_lines() if ligne.get_linestyle() == "-"]
        # Au moins 2 lignes de tendance (rouge + vert)
        self.assertGreaterEqual(len(lignes), 2)


class TestPredicton2050(unittest.TestCase):
    """Vérifie que la prédiction 2050 est raisonnable."""

    def setUp(self):
        self.df = load_data()
        slope, intercept, *_ = linregress(
            self.df["Year"], self.df["CSIRO Adjusted Sea Level"]
        )
        self.prediction_2050 = slope * 2050 + intercept

    def test_prediction_positive(self):
        self.assertGreater(self.prediction_2050, 0)

    def test_prediction_plus_grande_derniere_valeur(self):
        """La prédiction 2050 doit être supérieure à la dernière valeur mesurée."""
        derniere = self.df["CSIRO Adjusted Sea Level"].iloc[-1]
        self.assertGreater(self.prediction_2050, derniere)


class TestConformiteCorrigeOfficiel(unittest.TestCase):
    """Les assertions de `test_module.py`, telles que le correcteur les lance."""

    def setUp(self):
        self.ax = draw_plot()

    def tearDown(self):
        plt.close(self.ax.figure)

    def test_draw_plot_renvoie_un_axes(self):
        """Le correcteur fait `ax = draw_plot()` puis `ax.get_title()`.

        Renvoyer une Figure, comme le font les projets 03 et 04, ferait échouer
        toutes les assertions sur un `AttributeError`.
        """
        self.assertIsInstance(self.ax, plt.Axes)

    def test_appel_sans_argument(self):
        self.assertIsNotNone(draw_plot())

    def test_titre_et_labels(self):
        self.assertEqual(self.ax.get_title(), "Rise in Sea Level")
        self.assertEqual(self.ax.get_xlabel(), "Year")
        self.assertEqual(self.ax.get_ylabel(), "Sea Level (inches)")

    def test_xticks(self):
        """La plage va jusqu'à 2075 : les droites sont tracées jusqu'en 2050."""
        self.assertEqual(
            self.ax.get_xticks().tolist(),
            [
                1850.0,
                1875.0,
                1900.0,
                1925.0,
                1950.0,
                1975.0,
                2000.0,
                2025.0,
                2050.0,
                2075.0,
            ],
        )

    def test_nuage_de_points(self):
        points = self.ax.get_children()[0].get_offsets().data.tolist()
        self.assertEqual(len(points), 134)
        np.testing.assert_almost_equal(points[0], [1880.0, 0.0], 7)
        np.testing.assert_almost_equal(points[-1], [2013.0, 8.980314951], 7)

    def test_droites_de_regression(self):
        """Deux droites : 1880-2050 (171 points) et 2000-2050 (51 points)."""
        lignes = self.ax.get_lines()
        self.assertEqual(len(lignes), 2)

        globale = lignes[0].get_ydata().tolist()
        self.assertEqual(len(globale), 171)
        np.testing.assert_almost_equal(globale[0], -0.5421240249263661, 7)
        np.testing.assert_almost_equal(globale[-1], 10.175455257136548, 7)

        recente = lignes[1].get_ydata().tolist()
        self.assertEqual(len(recente), 51)
        np.testing.assert_almost_equal(recente[0], 7.06107985777146, 7)
        np.testing.assert_almost_equal(recente[-1], 15.382443524364874, 7)


class TestChargementEchec(unittest.TestCase):
    """Le chemin d'erreur du téléchargement."""

    def test_telechargement_impossible_arrete_le_programme(self):
        with tempfile.TemporaryDirectory() as tmp:
            cible = pathlib.Path(tmp) / "epa-sea-level.csv"
            with mock.patch(
                "sea_level_predictor.urllib.request.urlopen",
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
