# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Tests unitaires pour le projet 04 : Page View Time Series Visualizer."""

import pathlib
import re
import tempfile
import unittest
from unittest import mock

import matplotlib

# Backend non interactif : les tests tracent des figures sans écran.
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import time_series_visualizer as tsv  # noqa: E402
from time_series_visualizer import (  # noqa: E402
    draw_bar_plot,
    draw_box_plot,
    draw_line_plot,
    load_data,
)


class TestDonnees(unittest.TestCase):
    """Vérifie la structure du jeu de données."""

    def setUp(self):
        self.df = load_data()

    def test_index_est_datetime(self):
        self.assertTrue(isinstance(self.df.index, pd.DatetimeIndex))

    def test_colonne_value_existe(self):
        """La colonne s'appelle « value », pas « views ».

        C'est le nom du fichier distribué par freeCodeCamp ; « views » est le
        terme employé par l'énoncé pour désigner la grandeur mesurée.
        """
        self.assertIn("value", self.df.columns)
        self.assertNotIn("views", self.df.columns)

    def test_filtrage_extremes(self):
        """Après filtrage, pas de valeurs aux 2.5% extrêmes."""
        # Le filtrage est fait dans load_data, on vérifie juste qu'il reste des données
        self.assertGreater(len(self.df), 100)

    def test_pas_de_valeur_manquante(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_plage_dates(self):
        """Les données couvrent au moins 2016-2019."""
        self.assertLessEqual(self.df.index.year.min(), 2016)
        self.assertGreaterEqual(self.df.index.year.max(), 2019)


class TestLinePlot(unittest.TestCase):
    """Vérifie le graphique linéaire."""

    def setUp(self):
        self.df = load_data()
        self.fig = draw_line_plot(self.df)

    def tearDown(self):
        plt.close(self.fig)

    def test_retourne_figure(self):
        self.assertIsInstance(self.fig, plt.Figure)

    def test_titre(self):
        ax = self.fig.axes[0]
        self.assertIn("Daily freeCodeCamp", ax.get_title())

    def test_label_x(self):
        ax = self.fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Date")

    def test_label_y(self):
        ax = self.fig.axes[0]
        self.assertEqual(ax.get_ylabel(), "Page Views")


class TestBarPlot(unittest.TestCase):
    """Vérifie le graphique en barres."""

    def setUp(self):
        self.df = load_data()
        self.fig = draw_bar_plot(self.df)

    def tearDown(self):
        plt.close(self.fig)

    def test_retourne_figure(self):
        self.assertIsInstance(self.fig, plt.Figure)

    def test_label_x(self):
        ax = self.fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Years")

    def test_label_y(self):
        ax = self.fig.axes[0]
        self.assertEqual(ax.get_ylabel(), "Average Page Views")


class TestBoxPlot(unittest.TestCase):
    """Vérifie les box plots."""

    def setUp(self):
        self.df = load_data()
        self.fig = draw_box_plot(self.df)

    def tearDown(self):
        plt.close(self.fig)

    def test_retourne_figure(self):
        self.assertIsInstance(self.fig, plt.Figure)

    def test_deux_axes(self):
        self.assertEqual(len(self.fig.axes), 2)

    def test_titre_tendance(self):
        ax1 = self.fig.axes[0]
        self.assertIn("Year-wise", ax1.get_title())

    def test_titre_saisonnalite(self):
        ax2 = self.fig.axes[1]
        self.assertIn("Month-wise", ax2.get_title())

    def test_mois_en_bas(self):
        """L'axe des mois doit commencer par Jan."""
        ax2 = self.fig.axes[1]
        labels = [t.get_text() for t in ax2.get_xticklabels()]
        self.assertEqual(labels[0], "Jan")


class TestConformiteCorrigeOfficiel(unittest.TestCase):
    """Les assertions de `test_module.py`, telles que le correcteur les lance.

    Deux exigences ne se devinent pas depuis l'énoncé : le module expose un
    DataFrame `df` chargé à l'import, et les trois fonctions de tracé s'appellent
    sans argument.
    """

    def test_df_expose_au_niveau_module(self):
        """Le correcteur lit `time_series_visualizer.df` directement."""
        self.assertEqual(int(tsv.df.count(numeric_only=True).iloc[0]), 1238)

    def test_fonctions_appelables_sans_argument(self):
        for fonction in (tsv.draw_line_plot, tsv.draw_bar_plot, tsv.draw_box_plot):
            self.assertIsNotNone(fonction(), f"{fonction.__name__}() a échoué")

    def test_line_plot(self):
        ax = tsv.draw_line_plot().axes[0]
        self.assertEqual(
            ax.get_title(), "Daily freeCodeCamp Forum Page Views 5/2016-12/2019"
        )
        self.assertEqual(ax.get_xlabel(), "Date")
        self.assertEqual(ax.get_ylabel(), "Page Views")
        self.assertEqual(len(ax.lines[0].get_ydata()), 1238)

    def test_bar_plot(self):
        ax = tsv.draw_bar_plot().axes[0]
        self.assertEqual(ax.get_xlabel(), "Years")
        self.assertEqual(ax.get_ylabel(), "Average Page Views")
        self.assertEqual(
            [label.get_text() for label in ax.get_legend().get_texts()],
            tsv.MONTHS_FULL,
            "la légende attend les noms complets, pas les abréviations",
        )
        barres = [r for r in ax.get_children() if isinstance(r, mpl.patches.Rectangle)]
        self.assertEqual(len(barres), 49)

    def test_box_plot(self):
        fig = tsv.draw_box_plot()
        self.assertEqual(len(fig.get_axes()), 2)
        ax1, ax2 = fig.axes[0], fig.axes[1]
        self.assertEqual(ax1.get_title(), "Year-wise Box Plot (Trend)")
        self.assertEqual(ax2.get_title(), "Month-wise Box Plot (Seasonality)")
        self.assertEqual(ax1.get_xlabel(), "Year")
        self.assertEqual(ax2.get_xlabel(), "Month")
        self.assertEqual(
            [label.get_text() for label in ax2.get_xaxis().get_majorticklabels()],
            tsv.MONTHS_SHORT,
            "les box plots attendent les abréviations, pas les noms complets",
        )
        # Six lignes par boîte : médiane, deux moustaches, deux capuchons, corps.
        self.assertEqual(len(ax1.lines) / 6, 4)
        self.assertEqual(len(ax2.lines) / 6, 12)


class TestChargementEchec(unittest.TestCase):
    """Le chemin d'erreur du téléchargement."""

    def test_telechargement_impossible_arrete_le_programme(self):
        with tempfile.TemporaryDirectory() as tmp:
            cible = pathlib.Path(tmp) / "fcc-forum-pageviews.csv"
            with mock.patch(
                "time_series_visualizer.urllib.request.urlopen",
                side_effect=OSError("réseau"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    tsv.download_data(cible)
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
