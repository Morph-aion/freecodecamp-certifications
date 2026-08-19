"""Tests unitaires du projet Cat/Dog.

Contrainte de conception : **ces tests ne doivent pas importer TensorFlow.**
L'import prend plusieurs secondes et exige l'environnement complet, ce qui suffit
à ce qu'on cesse de lancer la suite. La logique testée ici est justement celle qui
n'a besoin d'aucun réseau : l'ordre des fichiers et la cohérence du corrigé.

Ce que ces tests ne couvrent pas, volontairement : l'architecture du modèle,
l'entraînement, la justesse obtenue. C'est lent, non déterministe, et l'énoncé
fournit déjà son propre critère d'évaluation.

Lancement :
    python -m unittest test_units
"""

import os
import re
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(
    BASE_DIR, "data", "raw", "cats_and_dogs", "test", "c_and_d"
)

# Corrigé officiel, dupliqué ici plutôt qu'importé de classifier.py : l'importer
# tirerait TensorFlow. La duplication est assumée et couverte par un test de
# cohérence (voir TestCoherenceAvecLeScript).
ANSWERS = [
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]

# Étiquettes établies en regardant les images, indépendamment de tout raisonnement
# sur les noms de fichiers. Ce sont les seuls faits qui permettent de trancher à
# quel ordre le corrigé se réfère.
GROUND_TRUTH_OBSERVEE = {
    "1.jpg": 1,  # chien noir et blanc tenu par une personne
    "2.jpg": 1,  # chien roux à collier rouge
    "10.jpg": 0,  # chat blanc et roux assis sur un tapis
}


class TestOrdreDuCorrige(unittest.TestCase):
    """À quel ordre `ANSWERS` se réfère-t-il ?

    Question piégeuse et décisive : `flow_from_directory` trie les fichiers
    lexicographiquement (`10.jpg` avant `2.jpg`), donc l'ordre du générateur
    diffère de l'ordre numérique. Se tromper d'hypothèse fait chuter le score à
    56 %, sous le seuil, y compris pour un modèle parfait.

    Ces tests fixent la réponse à partir du contenu réel des images, pour qu'une
    « correction » intuitive ne puisse pas réintroduire le décalage.
    """

    def test_les_deux_ordres_different_bien(self):
        """Sans cette différence, la question ne se poserait pas."""
        noms = [f"{i}.jpg" for i in range(1, 51)]
        self.assertNotEqual(sorted(noms), noms)
        self.assertEqual(sorted(noms)[1], "10.jpg")

    def test_le_corrige_suit_l_ordre_du_generateur(self):
        """Preuve par les images : ANSWERS[1] = 0 décrit 10.jpg, un chat.

        Si le corrigé suivait l'ordre numérique, ANSWERS[1] décrirait 2.jpg, qui
        est un chien : la valeur serait 1, pas 0.
        """
        ordre_generateur = sorted(f"{i}.jpg" for i in range(1, 51))
        for position, nom in enumerate(ordre_generateur):
            if nom in GROUND_TRUTH_OBSERVEE:
                with self.subTest(fichier=nom, position=position):
                    self.assertEqual(
                        ANSWERS[position],
                        GROUND_TRUTH_OBSERVEE[nom],
                        f"{nom} est {'un chien' if GROUND_TRUTH_OBSERVEE[nom] else 'un chat'}",
                    )

    def test_l_ordre_numerique_contredit_les_images(self):
        """Contre-épreuve : l'hypothèse inverse est incompatible avec 2.jpg."""
        ordre_numerique = [f"{i}.jpg" for i in range(1, 51)]
        position_de_2 = ordre_numerique.index("2.jpg")
        self.assertNotEqual(
            ANSWERS[position_de_2],
            GROUND_TRUTH_OBSERVEE["2.jpg"],
            "si cette assertion échoue, l'hypothèse de l'ordre numérique redevient "
            "plausible et tout ce module est à revoir",
        )

    def test_reordonner_ferait_chuter_un_modele_parfait_sous_le_seuil(self):
        """Chiffre le coût de la « correction » qu'il ne faut pas faire.

        Un modèle parfait prédit correctement chaque image dans l'ordre du
        générateur. Réordonner ces probabilités par numéro de fichier avant de les
        comparer à ANSWERS ferait tomber la concordance à 28/50, soit 56 %.
        """
        ordre_generateur = sorted(f"{i}.jpg" for i in range(1, 51))
        # Modèle parfait : sa sortie EST le corrigé, dans l'ordre du générateur.
        probabilites = list(ANSWERS)

        # La « correction » erronée : réordonner par numéro de fichier.
        numeros = [int(re.search(r"\d+", nom).group()) for nom in ordre_generateur]
        reordonnees = [p for _, p in sorted(zip(numeros, probabilites, strict=True))]

        concordance = sum(
            1 for p, a in zip(reordonnees, ANSWERS, strict=True) if round(p) == a
        )
        self.assertEqual(concordance, 28)
        self.assertLess(concordance / len(ANSWERS), 0.63)

    def test_sans_reordonner_un_modele_parfait_reussit(self):
        """Comparaison directe, telle que la fait `evaluate`."""
        probabilites = [float(a) for a in ANSWERS]
        concordance = sum(
            1 for p, a in zip(probabilites, ANSWERS, strict=True) if round(p) == a
        )
        self.assertEqual(concordance, 50)


class TestCorrige(unittest.TestCase):
    """Cohérence interne du corrigé officiel."""

    def test_cinquante_reponses(self):
        self.assertEqual(len(ANSWERS), 50)

    def test_uniquement_des_binaires(self):
        self.assertTrue(all(answer in (0, 1) for answer in ANSWERS))

    def test_les_deux_classes_sont_representees(self):
        chiens = sum(ANSWERS)
        self.assertGreater(chiens, 15)
        self.assertLess(chiens, 35)

    def test_predire_toujours_la_meme_classe_ne_suffit_pas(self):
        """Garde-fou : le seuil doit exiger un vrai classifieur."""
        chiens = sum(ANSWERS)
        meilleure_classe_constante = max(chiens, len(ANSWERS) - chiens)
        self.assertLess(meilleure_classe_constante / len(ANSWERS), 0.63)


class TestJeuDeDonnees(unittest.TestCase):
    """Structure attendue sur le disque, si les données sont présentes.

    Ignorés proprement si le jeu n'a pas encore été téléchargé, pour que la suite
    reste lançable sur une copie fraîche du dépôt.
    """

    def setUp(self):
        if not os.path.isdir(TEST_IMAGES_DIR):
            self.skipTest("jeu de données absent, lancer classifier.py d'abord")

    def test_cinquante_images_de_test(self):
        images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".jpg")]
        self.assertEqual(len(images), 50)

    def test_les_images_sont_numerotees_de_un_a_cinquante(self):
        numeros = sorted(
            int(re.search(r"\d+", f).group())
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.endswith(".jpg")
        )
        self.assertEqual(numeros, list(range(1, 51)))

    def test_les_images_de_reference_existent(self):
        """Les trois fichiers sur lesquels repose la preuve d'ordre."""
        for nom in GROUND_TRUTH_OBSERVEE:
            with self.subTest(fichier=nom):
                self.assertTrue(os.path.isfile(os.path.join(TEST_IMAGES_DIR, nom)))

    def test_les_repertoires_train_et_validation_sont_complets(self):
        racine = os.path.join(BASE_DIR, "data", "raw", "cats_and_dogs")
        attendus = {
            ("train", "cats"): 1000,
            ("train", "dogs"): 1000,
            ("validation", "cats"): 500,
            ("validation", "dogs"): 500,
        }
        for (split, classe), compte in attendus.items():
            chemin = os.path.join(racine, split, classe)
            with self.subTest(split=split, classe=classe):
                self.assertTrue(os.path.isdir(chemin), f"{chemin} manquant")
                self.assertEqual(len(os.listdir(chemin)), compte)


class TestCoherenceAvecLeScript(unittest.TestCase):
    """`ANSWERS` est dupliqué dans ce module pour éviter d'importer TensorFlow.

    Ce test lit `classifier.py` comme du texte et compare les deux copies, pour
    que la duplication ne puisse pas diverger en silence.
    """

    def test_le_corrige_est_identique_a_celui_du_script(self):
        import ast

        with open(os.path.join(BASE_DIR, "classifier.py"), encoding="utf-8") as f:
            source = f.read()
        arbre = ast.parse(source)
        for noeud in arbre.body:
            if isinstance(noeud, ast.Assign) and noeud.targets[0].id == "ANSWERS":
                self.assertEqual(ast.literal_eval(noeud.value), ANSWERS)
                return
        self.fail("ANSWERS introuvable dans classifier.py")

    def test_predict_ne_reordonne_pas(self):
        """Régression : une version antérieure réordonnait les probabilités.

        Vérifie que `predict` ne contient plus de tri par numéro de fichier, la
        « correction » qui ferait chuter le score à 56 %.
        """
        with open(os.path.join(BASE_DIR, "classifier.py"), encoding="utf-8") as f:
            source = f.read()
        corps_predict = source.split("def predict(")[1].split("\ndef ")[0]
        code = "\n".join(
            ligne
            for ligne in corps_predict.splitlines()
            if not ligne.strip().startswith("#")
        )
        # La docstring mentionne le sujet, on ne regarde donc que le code exécuté.
        code_execute = code.split('"""')[-1]
        self.assertNotIn("sorted(", code_execute)


class TestReferencesDeLaDocumentation(unittest.TestCase):
    """`notions-mobilisees.md` cite des lignes précises de `classifier.py`.

    Ces références se périment silencieusement dès qu'on ajoute du code au-dessus
    d'elles, ce qui est arrivé deux fois. Ce test attache chaque numéro cité à ce
    qu'il est censé désigner, et échoue au premier décalage.

    Ajouter une entrée ici à chaque nouvelle référence dans le document.
    """

    ANCRES = {
        93: "RANDOM_SEED = 42",
        98: "EPOCHS = 30",
        251: "rescale",
        257: "horizontal_flip",
        312: "Conv2D(16",
        313: "MaxPooling2D",
        319: "MaxPooling2D",
        321: "Dropout(0.5)",
        322: "Dense(512",
        323: 'Dense(1, activation="sigmoid")',
        327: "optimizer=",
        328: "binary_crossentropy",
        329: "metrics=",
        352: "steps_per_epoch =",
        353: "validation_steps =",
        514: "def save_history_plots",
        597: "round(probability)",
        616: "np.random.seed",
        617: "tf.random.set_seed",
    }

    def setUp(self):
        with open(os.path.join(BASE_DIR, "classifier.py"), encoding="utf-8") as f:
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
        """Aucune référence du document ne doit échapper au test ci-dessus."""
        chemin = os.path.join(BASE_DIR, "docs", "notions-mobilisees.md")
        with open(chemin, encoding="utf-8") as f:
            document = f.read()

        citees = set()
        for debut, fin in re.findall(r"classifier\.py#L(\d+)(?:-L(\d+))?", document):
            citees.add(int(debut))
            if fin:
                citees.add(int(fin))

        non_couvertes = citees - set(self.ANCRES)
        self.assertEqual(
            non_couvertes,
            set(),
            f"lignes citées sans ancre de vérification : {sorted(non_couvertes)}",
        )


if __name__ == "__main__":
    unittest.main()
