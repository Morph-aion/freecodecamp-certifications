# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "scikit-learn",
# ]
# ///
"""Tests unitaires pour le projet 04 : Health Costs Regression."""

import pathlib
import tempfile
import unittest
from unittest import mock

import pandas as pd
from regression import (
    TARGET_CANDIDATES,
    add_interaction,
    download_data,
    evaluate_model,
    load_data,
    plot_predictions,
    prepare_features,
    resolve_target,
    split_data,
    train_model,
)
from sklearn.linear_model import LinearRegression


class TestDonnees(unittest.TestCase):
    """Vérifie la structure du jeu de données."""

    def setUp(self):
        self.df = load_data()
        self.target = resolve_target(self.df)

    def test_nombre_lignes(self):
        self.assertEqual(len(self.df), 1338)

    def test_colonnes_presentes(self):
        # La cible s'appelle « expenses » dans le jeu officiel freeCodeCamp et
        # « charges » dans la version Kaggle : les deux sont acceptées.
        attendues = {"age", "sex", "bmi", "children", "smoker", "region"}
        self.assertEqual(set(self.df.columns) - {self.target}, attendues)
        self.assertIn(self.target, TARGET_CANDIDATES)

    def test_aucune_valeur_manquante(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_types(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df["age"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df["bmi"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df["children"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df[self.target]))


class TestEncodage(unittest.TestCase):
    """Vérifie la conversion des catégorielles en nombres."""

    def setUp(self):
        self.df = load_data()
        self.encoded = prepare_features(self.df)

    def test_colonnes_catégorielles_disparues(self):
        for col in ["sex", "smoker", "region"]:
            self.assertNotIn(col, self.encoded.columns)

    def test_nouvelles_colonnes_créées(self):
        attendues = {
            "sex_male",
            "smoker_yes",
            "region_northwest",
            "region_southeast",
            "region_southwest",
        }
        self.assertTrue(attendues.issubset(set(self.encoded.columns)))

    def test_colonnes_encodees_sont_binaires(self):
        """Distingue le one-hot d'un encodage label, qui produirait un ordre.

        Le nom précédent (« pas de label encoding ») promettait plus que le test
        ne vérifiait : constater la présence de `sex_male` n'exclut pas qu'un
        `LabelEncoder` ait été appliqué avant. Ce qui distingue réellement les
        deux encodages, c'est l'ensemble des valeurs : one-hot ne produit que
        des 0 et des 1, un encodage label sur `region` produirait 0, 1, 2, 3.
        """
        for col in self.encoded.columns:
            if col.startswith(("sex_", "smoker_", "region_")):
                valeurs = set(self.encoded[col].astype(int).unique())
                self.assertTrue(
                    valeurs.issubset({0, 1}),
                    f"{col} contient {valeurs}, ce n'est pas un encodage binaire",
                )

    def test_drop_first_colinéarité(self):
        """Vérifie qu'une colonne par catégorie est supprimée (evite multicolinéarité)."""
        # sex: 2 catégories → 1 colonne après drop_first
        self.assertIn("sex_male", self.encoded.columns)
        # smoker: 2 catégories → 1 colonne
        self.assertIn("smoker_yes", self.encoded.columns)
        # region: 4 catégories → 3 colonnes
        region_cols = [c for c in self.encoded.columns if c.startswith("region_")]
        self.assertEqual(len(region_cols), 3)


class TestInteraction(unittest.TestCase):
    """Vérifie la feature d'interaction bmi × smoker."""

    def setUp(self):
        self.df = load_data()
        self.encoded = prepare_features(self.df)
        self.with_interaction = add_interaction(self.encoded)

    def test_colonne_créée(self):
        self.assertIn("bmi_smoker", self.with_interaction.columns)

    def test_valeurs_correctes(self):
        mask = self.with_interaction["smoker_yes"] == 1
        expected = self.with_interaction.loc[mask, "bmi"] * 1
        actual = self.with_interaction.loc[mask, "bmi_smoker"]
        pd.testing.assert_series_equal(expected, actual, check_names=False)

    def test_zeros_pour_non_fumeurs(self):
        mask = self.with_interaction["smoker_yes"] == 0
        self.assertTrue((self.with_interaction.loc[mask, "bmi_smoker"] == 0).all())

    def test_echec_explicite_si_colonne_absente(self):
        """Sans smoker_yes, l'absence d'interaction doit lever, pas passer."""
        # Un échec silencieux ferait chuter la MAE à ~4200 et raterait le seuil
        # sans indiquer pourquoi : le cas doit être bruyant.
        brut = load_data()
        with self.assertRaises(KeyError):
            add_interaction(brut)

    def test_message_erreur_nomme_les_colonnes(self):
        brut = load_data()
        with self.assertRaises(KeyError) as ctx:
            add_interaction(brut)
        self.assertIn("smoker_yes", str(ctx.exception))
        self.assertIn("prepare_features", str(ctx.exception))


class TestChargement(unittest.TestCase):
    """Chemins d'erreur du chargement, seul code non couvert par ailleurs."""

    def test_telechargement_impossible_arrete_le_programme(self):
        """Une panne réseau doit arrêter net, pas laisser continuer sans données."""
        with tempfile.TemporaryDirectory() as tmp:
            cible = pathlib.Path(tmp) / "insurance.csv"
            with mock.patch(
                "regression.urllib.request.urlopen", side_effect=OSError("réseau")
            ):
                with self.assertRaises(SystemExit) as ctx:
                    download_data(cible)
        self.assertEqual(ctx.exception.code, 1)

    def test_cible_absente_leve_une_erreur_nommant_les_candidats(self):
        sans_cible = load_data().drop(columns=[resolve_target(load_data())])
        with self.assertRaises(KeyError) as ctx:
            resolve_target(sans_cible)
        for nom in TARGET_CANDIDATES:
            self.assertIn(nom, str(ctx.exception))


class TestFigure(unittest.TestCase):
    """La figure est un livrable du pipeline : son absence doit se voir."""

    def test_le_fichier_est_un_png_non_vide(self):
        df = add_interaction(prepare_features(load_data()))
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)
        y_pred = evaluate_model(model, X_test, y_test)["y_pred"]

        with tempfile.TemporaryDirectory() as tmp:
            sortie = pathlib.Path(tmp) / "sous-dossier" / "figure.png"
            plot_predictions(y_test, y_pred, sortie)
            self.assertTrue(sortie.exists(), "figure non créée")
            self.assertGreater(sortie.stat().st_size, 1000, "figure suspecte")
            # Signature PNG : les huit premiers octets du format.
            self.assertEqual(sortie.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class TestSplit(unittest.TestCase):
    """Vérifie la répartition entraînement/test."""

    def setUp(self):
        self.df = add_interaction(prepare_features(load_data()))
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

    def test_proportions(self):
        total = len(self.X_train) + len(self.X_test)
        ratio = len(self.X_test) / total
        self.assertAlmostEqual(ratio, 0.2, places=2)

    def test_features_identiques(self):
        self.assertEqual(list(self.X_train.columns), list(self.X_test.columns))

    def test_target_absente_des_features(self):
        for name in TARGET_CANDIDATES:
            self.assertNotIn(name, self.X_train.columns)
            self.assertNotIn(name, self.X_test.columns)


class TestModele(unittest.TestCase):
    """Vérifie l'entraînement et l'évaluation du modèle."""

    def setUp(self):
        self.df = add_interaction(prepare_features(load_data()))
        X_train, X_test, y_train, y_test = split_data(self.df)
        self.model = train_model(X_train, y_train)
        self.metrics = evaluate_model(self.model, X_test, y_test)

    def test_model_type(self):
        self.assertIsInstance(self.model, LinearRegression)

    def test_mae_inferieure_seuil(self):
        self.assertLess(self.metrics["mae"], 3500)

    def test_r2_proche_de_la_valeur_mesuree(self):
        """Seuil serré : 0,5 ne détecterait aucune régression réelle.

        Le modèle atteint 0,865. Perdre l'interaction, le pire accident
        plausible de ce pipeline, donne encore 0,784 : un seuil de 0,5 laisserait
        passer cette régression sans rien signaler.
        """
        self.assertGreater(self.metrics["r2"], 0.82)

    def test_metriques_coherentes_entre_elles(self):
        """RMSE doit être la racine de MSE, et la MAE inférieure à la RMSE."""
        self.assertAlmostEqual(
            self.metrics["rmse"], self.metrics["mse"] ** 0.5, places=4
        )
        # Inégalité de Jensen : |erreur| moyenne ≤ racine de la moyenne des carrés.
        self.assertLessEqual(self.metrics["mae"], self.metrics["rmse"])


class TestSansInteraction(unittest.TestCase):
    """Comparaison : MAE avec et sans feature d'interaction."""

    def test_interaction_améliore(self):
        df = prepare_features(load_data())
        X_train, X_test, y_train, y_test = split_data(df)
        model_sans = train_model(X_train, y_train)
        mae_sans = evaluate_model(model_sans, X_test, y_test)["mae"]

        df_int = add_interaction(df)
        X_train_i, X_test_i, y_train_i, y_test_i = split_data(df_int)
        model_avec = train_model(X_train_i, y_train_i)
        mae_avec = evaluate_model(model_avec, X_test_i, y_test_i)["mae"]

        self.assertLess(
            mae_avec,
            mae_sans,
            f"L'interaction devrait réduire la MAE : {mae_avec:.0f} >= {mae_sans:.0f}",
        )


if __name__ == "__main__":
    unittest.main()
