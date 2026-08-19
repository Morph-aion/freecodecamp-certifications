# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
#     "tensorflow",
# ]
# ///
"""Tests unitaires pour le projet 05 : SMS Text Classifier.

Les tests de contrat entraînent un vrai modèle (quelques epochs) plutôt que
d'appeler un stub : c'est la seule façon de détecter les ruptures d'API Keras,
comme le refus des pandas.Series de chaînes en entrée de fit(). Le modèle
partagé est entraîné une fois pour toute la classe de tests.
"""

import pathlib
import unittest

import numpy as np
import pandas as pd
from classifier import (
    TEST_ANSWERS,
    TEST_MESSAGES,
    as_text_array,
    build_model,
    build_vectorizer,
    encode_labels,
    evaluate_model,
    load_data,
    predict_message,
    train_model,
)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"


class TestDonnees(unittest.TestCase):
    """Vérifie la structure des fichiers TSV."""

    def setUp(self):
        self.train_path = DATA_DIR / "train-data.tsv"
        self.valid_path = DATA_DIR / "valid-data.tsv"

    def test_fichiers_existent(self):
        self.assertTrue(
            self.train_path.exists(), f"Fichier non trouvé : {self.train_path}"
        )
        self.assertTrue(
            self.valid_path.exists(), f"Fichier non trouvé : {self.valid_path}"
        )

    def test_format_tsv_sans_header(self):
        """Les fichiers sont TSV sans ligne d'en-tête."""
        train = pd.read_csv(self.train_path, sep="\t", header=None)
        valid = pd.read_csv(self.valid_path, sep="\t", header=None)
        self.assertEqual(train.shape[1], 2)
        self.assertEqual(valid.shape[1], 2)

    def test_colonne_type_valide(self):
        """La colonne 0 ne contient que 'ham' ou 'spam'."""
        for path in [self.train_path, self.valid_path]:
            df = pd.read_csv(path, sep="\t", header=None)
            types = set(df[0].unique())
            self.assertTrue(
                types.issubset({"ham", "spam"}),
                f"Types inattendus dans {path.name} : {types - {'ham', 'spam'}}",
            )

    def test_nombre_lignes(self):
        train = pd.read_csv(self.train_path, sep="\t", header=None)
        valid = pd.read_csv(self.valid_path, sep="\t", header=None)
        self.assertGreater(len(train), 3000)
        self.assertGreater(len(valid), 1000)

    def test_déséquilibre_de_classes(self):
        """Le jeu est déséquilibré : 86,6 % de ham.

        Bornes serrées autour de la valeur mesurée plutôt que l'intervalle
        [0,7 ; 0,95] d'origine : celui-ci vérifiait seulement que le jeu est
        déséquilibré, pas qu'il s'agit du bon jeu. C'est ce taux exact que le
        README et les documents citent comme référence « toujours ham ».
        """
        train = pd.read_csv(self.train_path, sep="\t", header=None)
        ham_ratio = (train[0] == "ham").mean()
        self.assertAlmostEqual(ham_ratio, 0.866, places=2)


class TestConversionTexte(unittest.TestCase):
    """Vérifie as_text_array, garde-fou contre la rupture d'API Keras 3."""

    def test_series_pandas_devient_object(self):
        serie = pd.Series(["bonjour", "salut"])
        self.assertEqual(as_text_array(serie).dtype, object)

    def test_liste_devient_object(self):
        self.assertEqual(as_text_array(["a", "b"]).dtype, object)

    def test_dtype_unicode_refuse_par_keras_est_converti(self):
        """Un tableau <U… doit devenir object : Keras 3 rejette le premier."""
        unicode_array = np.array(["a", "b"])
        self.assertNotEqual(unicode_array.dtype, object)
        self.assertEqual(as_text_array(unicode_array).dtype, object)


class TestModeleEntraine(unittest.TestCase):
    """Entraîne un vrai modèle et vérifie le contrat de predict_message.

    3 epochs suffisent : on teste le contrat et l'absence de régression d'API,
    pas la performance finale (mesurée par classifier.py sur 30 epochs).
    """

    @classmethod
    def setUpClass(cls):
        from tensorflow import keras

        keras.utils.set_random_seed(42)
        train_df, valid_df = load_data()
        cls.valid_df = valid_df
        cls.valid_labels = encode_labels(valid_df)
        vectorizer = build_vectorizer(train_df["text"])
        cls.model = build_model(vectorizer)
        train_model(
            cls.model,
            train_df["text"],
            encode_labels(train_df),
            valid_df["text"],
            cls.valid_labels,
            epochs=3,
            verbose=0,
        )

    def test_le_modele_est_construit_et_entraine(self):
        """Le modèle a des poids ET a réellement appris.

        Le nom précédent (« entraînement sur Series pandas ») décrivait la
        motivation, pas l'assertion : un modèle jamais entraîné a lui aussi des
        paramètres. Ce que ce test vérifie, c'est que `fit()` a produit des
        prédictions non dégénérées : le refus des Series par Keras 3 ferait
        échouer setUpClass avant d'arriver ici.
        """
        self.assertGreater(self.model.count_params(), 0)
        # Un modèle non entraîné produit des probabilités presque identiques
        # partout : leur écart-type sur le jeu de validation vaut 0,001 avant
        # entraînement contre 0,187 après trois epochs. C'est cette dispersion,
        # et non un couple de messages choisis, qui atteste d'un apprentissage.
        probabilites = self.model.predict(
            as_text_array(self.valid_df["text"]), verbose=0
        ).ravel()
        self.assertGreater(probabilites.std(), 0.05)

    def test_retourne_une_liste_de_deux(self):
        result = predict_message(self.model, "how are you doing today")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_probabilité_entre_0_et_1(self):
        prob = predict_message(self.model, "hello there")[0]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_label_coherent_avec_la_probabilité(self):
        """Le label doit suivre le seuil 0.5, et 1 doit signifier spam."""
        for message in ("how are you doing today", "you have won £1000 cash! call now"):
            prob, label = predict_message(self.model, message)
            self.assertEqual(label, "spam" if prob >= 0.5 else "ham")

    def test_sens_des_classes(self):
        """0 = ham, 1 = spam : un spam évident doit donner une proba haute."""
        prob_spam = predict_message(self.model, "win a free cash prize call now")[0]
        prob_ham = predict_message(self.model, "ok see you tomorrow at home")[0]
        self.assertGreater(prob_spam, prob_ham)

    def test_bat_nettement_la_reference_toujours_ham(self):
        """Battre 0,866 de justesse ne suffit pas à prouver un apprentissage.

        Un modèle qui ne détecterait qu'une poignée de spam dépasserait déjà la
        référence. Le modèle à 3 epochs atteint 0,935 : le seuil est placé juste
        en dessous pour rester discriminant.
        """
        metrics = evaluate_model(self.model, self.valid_df["text"], self.valid_labels)
        reference = 1 - self.valid_labels.mean()
        self.assertGreater(metrics["accuracy"], reference)
        self.assertGreater(metrics["accuracy"], 0.90)

    def test_rappel_spam_suffisant(self):
        """La métrique qui distingue un vrai classifieur d'un « toujours ham ».

        Mesuré à 0,658 après 3 epochs (0,936 après les 30 du pipeline complet).
        """
        metrics = evaluate_model(self.model, self.valid_df["text"], self.valid_labels)
        self.assertGreater(metrics["recall_spam"], 0.55)

    def test_matrice_de_confusion_coherente(self):
        """La confusion doit totaliser le jeu de validation et refléter la justesse."""
        metrics = evaluate_model(self.model, self.valid_df["text"], self.valid_labels)
        (tn, fp), (fn, tp) = metrics["confusion"]
        self.assertEqual(tn + fp + fn + tp, len(self.valid_labels))
        self.assertAlmostEqual(
            metrics["accuracy"], (tn + tp) / len(self.valid_labels), places=6
        )


class TestTestOfficiel(unittest.TestCase):
    """Vérifie la cohérence de la liste des 7 messages du test officiel."""

    def test_correspondance_message_réponse(self):
        self.assertEqual(len(TEST_MESSAGES), 7)
        self.assertEqual(len(TEST_ANSWERS), 7)

    def test_réponses_valides(self):
        for ans in TEST_ANSWERS:
            self.assertIn(ans, ("ham", "spam"))

    def test_messages_absents_du_corpus(self):
        """Les 7 messages ne doivent pas être dans les données d'entraînement.

        S'ils y étaient, le test officiel mesurerait de la mémorisation.
        """
        train_df, valid_df = load_data()
        corpus = set(
            pd.concat([train_df["text"], valid_df["text"]]).str.strip().str.lower()
        )
        for message in TEST_MESSAGES:
            self.assertNotIn(message.strip().lower(), corpus)


if __name__ == "__main__":
    unittest.main()
