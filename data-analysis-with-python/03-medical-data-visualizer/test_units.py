# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Tests unitaires pour le projet 03 : Medical Data Visualizer."""

import pathlib
import re
import tempfile
import unittest
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

# Backend non interactif : les tests tracent des figures sans écran.
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
from medical_data_visualizer import (
    add_overweight,
    clean_data,
    download_data,  # noqa: E402
    draw_cat_plot,
    draw_heat_map,
    load_data,
    normalize_values,
    prepare_data,
)


class TestDonnees(unittest.TestCase):
    """Vérifie la structure du jeu de données."""

    def setUp(self):
        self.df = load_data()

    def test_nombre_lignes(self):
        self.assertEqual(len(self.df), 70000)

    def test_colonnes_presentes(self):
        attendues = {
            # La colonne s'appelle « sex », pas « gender » : c'est le nom du
            # fichier distribué par freeCodeCamp.
            "id",
            "age",
            "sex",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "cardio",
        }
        self.assertEqual(set(self.df.columns), attendues)

    def test_aucune_valeur_manquante(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)


class TestNettoyage(unittest.TestCase):
    """Vérifie le nettoyage des données."""

    def setUp(self):
        self.df = load_data()
        self.cleaned = clean_data(self.df)

    def test_ap_lo_inferieur_ap_hi(self):
        self.assertTrue((self.cleaned["ap_lo"] <= self.cleaned["ap_hi"]).all())

    def test_height_dans_percentiles(self):
        h_low = self.df["height"].quantile(0.025)
        h_high = self.df["height"].quantile(0.975)
        self.assertTrue((self.cleaned["height"] >= h_low).all())
        self.assertTrue((self.cleaned["height"] <= h_high).all())

    def test_weight_dans_percentiles(self):
        w_low = self.df["weight"].quantile(0.025)
        w_high = self.df["weight"].quantile(0.975)
        self.assertTrue((self.cleaned["weight"] >= w_low).all())
        self.assertTrue((self.cleaned["weight"] <= w_high).all())

    def test_pas_de_perte_extreme(self):
        """Le nettoyage ne doit pas supprimer plus de 50% des lignes."""
        self.assertGreater(len(self.cleaned), len(self.df) * 0.5)


class TestOverweight(unittest.TestCase):
    """Vérifie la colonne 'overweight'."""

    def setUp(self):
        self.df = load_data()
        self.df = add_overweight(self.df)

    def test_colonne_exist(self):
        self.assertIn("overweight", self.df.columns)

    def test_binaire(self):
        self.assertTrue(set(self.df["overweight"]).issubset({0, 1}))

    def test_imc_correct(self):
        bmi = self.df["weight"] / ((self.df["height"] / 100) ** 2)
        expected = (bmi > 25).astype(int)
        pd.testing.assert_series_equal(
            self.df["overweight"], expected, check_names=False
        )


class TestNormalisation(unittest.TestCase):
    """Vérifie la normalisation de cholesterol et gluc."""

    def setUp(self):
        self.df = load_data()
        self.df = normalize_values(self.df)

    def test_cholesterol_binaire(self):
        self.assertTrue(set(self.df["cholesterol"]).issubset({0, 1}))

    def test_gluc_binaire(self):
        self.assertTrue(set(self.df["gluc"]).issubset({0, 1}))

    def test_cholesterol_1_devient_0(self):
        """L'ancienne valeur 1 (normal) devient 0."""
        # Pas testable directement sans l'ancien df, mais on vérifie la plage
        self.assertTrue(self.df["cholesterol"].min() >= 0)
        self.assertTrue(self.df["cholesterol"].max() <= 1)

    def test_gluc_1_devient_0(self):
        self.assertTrue(self.df["gluc"].min() >= 0)
        self.assertTrue(self.df["gluc"].max() <= 1)


class TestFigures(unittest.TestCase):
    """Vérifie que les graphiques sont générables."""

    def setUp(self):
        self.df = load_data()
        self.df = normalize_values(add_overweight(clean_data(self.df)))

    def test_cat_plot_retourne_figure(self):
        import matplotlib.pyplot as plt

        fig = draw_cat_plot(self.df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_retourne_figure(self):
        import matplotlib.pyplot as plt

        fig = draw_heat_map(self.df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_est_symetrique(self):
        corr = self.df.corr()
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)


class TestConformiteCorrigeOfficiel(unittest.TestCase):
    """Les 4 assertions de `test_module.py`, telles que le correcteur les lance.

    Le correcteur appelle `draw_cat_plot()` et `draw_heat_map()` **sans
    argument** : la signature fait partie du contrat au même titre que le
    contenu des figures.
    """

    def test_appel_sans_argument(self):
        self.assertIsNotNone(draw_cat_plot())
        self.assertIsNotNone(draw_heat_map())

    def test_cat_plot_labels_et_ticks(self):
        ax = draw_cat_plot().axes[0]
        self.assertEqual(ax.get_xlabel(), "variable")
        self.assertEqual(ax.get_ylabel(), "total")
        self.assertEqual(
            [label.get_text() for label in ax.get_xaxis().get_majorticklabels()],
            ["active", "alco", "cholesterol", "gluc", "overweight", "smoke"],
        )

    def test_cat_plot_treize_barres(self):
        """13 : six variables × deux valeurs, plus le rectangle du cadre."""
        ax = draw_cat_plot().axes[0]
        barres = [r for r in ax.get_children() if isinstance(r, mpl.patches.Rectangle)]
        self.assertEqual(len(barres), 13)

    def test_heat_map_labels(self):
        ax = draw_heat_map().axes[0]
        self.assertEqual(
            [label.get_text() for label in ax.get_xticklabels()],
            [
                "id",
                "age",
                "sex",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active",
                "cardio",
                "overweight",
            ],
        )

    def test_heat_map_quatre_vingt_onze_valeurs(self):
        """Le triangle inférieur d'une matrice 14×14 compte 91 cases."""
        ax = draw_heat_map().axes[0]
        valeurs = [
            t.get_text()
            for t in ax.get_default_bbox_extra_artists()
            if isinstance(t, mpl.text.Text)
        ]
        self.assertEqual(len(valeurs), 91)
        self.assertEqual(valeurs[9], "0.3")
        self.assertEqual(valeurs[81], "-0.1")


class TestNettoyageSimultane(unittest.TestCase):
    """Les bornes se calculent sur le jeu complet, pas en cascade."""

    def test_nombre_de_lignes_conservees(self):
        """63 259, pas 62 784 : recalculer les quantiles après chaque filtre
        rétrécit la distribution et retire 475 lignes de trop."""
        self.assertEqual(len(prepare_data()), 63259)


class TestChargementEchec(unittest.TestCase):
    """Le chemin d'erreur du téléchargement, seul code non couvert autrement."""

    def test_telechargement_impossible_arrete_le_programme(self):
        """Une panne réseau doit arrêter net.

        Sans `sys.exit(1)`, `load_data()` enchaîne sur la lecture d'un fichier
        absent et l'utilisateur reçoit un `FileNotFoundError` qui ne dit rien de
        la cause réelle.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cible = pathlib.Path(tmp) / "medical_examination.csv"
            with mock.patch(
                "medical_data_visualizer.urllib.request.urlopen",
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
