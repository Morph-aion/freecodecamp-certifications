# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""Tests unitaires pour le projet 01 : Mean-Variance-Standard Deviation Calculator."""

import pathlib
import re
import unittest

from mean_var_std import calculate


class TestCalculate(unittest.TestCase):
    """Vérifie le fonctionnement de calculate()."""

    def test_resultat_complet(self):
        result = calculate([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            result,
            {
                "mean": [[3.0, 4.0, 5.0], [1.0, 4.0, 7.0], 4.0],
                # Les colonnes de cette matrice sont [0,3,6], [1,4,7], [2,5,8] :
                # écartées de 3, donc de variance 6. Les lignes sont [0,1,2],
                # [3,4,5], [6,7,8] : écartées de 1, donc de variance 2/3. Les
                # deux axes n'ont aucune raison de coïncider.
                "variance": [
                    [6.0, 6.0, 6.0],
                    [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
                    6.666666666666667,
                ],
                "standard deviation": [
                    [2.449489742783178, 2.449489742783178, 2.449489742783178],
                    [0.816496580927726, 0.816496580927726, 0.816496580927726],
                    2.581988897471611,
                ],
                "max": [[6, 7, 8], [2, 5, 8], 8],
                "min": [[0, 1, 2], [0, 3, 6], 0],
                "sum": [[9, 12, 15], [3, 12, 21], 36],
            },
        )

    def test_liste_trop_courte(self):
        with self.assertRaisesRegex(ValueError, "List must contain nine numbers."):
            calculate([1, 2, 3, 4, 5])

    def test_liste_trop_longue(self):
        with self.assertRaisesRegex(ValueError, "List must contain nine numbers."):
            calculate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_liste_vide(self):
        with self.assertRaisesRegex(ValueError, "List must contain nine numbers."):
            calculate([])

    def test_variance_de_population_pas_echantillon(self):
        """NumPy divise par n, pandas par n-1 : le correcteur attend NumPy.

        Sur la ligne [0, 1, 2], ddof=0 donne 0,667 et ddof=1 donne 1,0. Utiliser
        pandas ou forcer ddof=1 ferait échouer la cellule officielle sans que le
        calcul soit faux pour autant : il répondrait à une autre question.
        """
        result = calculate([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(result["variance"][1][0], 2 / 3, places=12)
        self.assertNotAlmostEqual(result["variance"][1][0], 1.0, places=6)

    def test_conforme_au_corrige_officiel(self):
        """Le premier jeu de `test_module.py`, celui que le correcteur exécute.

        La suite [0..8] du test précédent est trompeuse : ses colonnes sont
        régulièrement espacées, ce qui masquerait certaines erreurs d'axe.
        """
        result = calculate([2, 6, 2, 8, 4, 0, 1, 5, 7])
        self.assertEqual(result["sum"], [[11, 15, 9], [10, 12, 13], 35])
        self.assertEqual(result["max"], [[8, 6, 7], [6, 8, 7], 8])
        self.assertAlmostEqual(result["mean"][2], 3.888888888888889, places=12)
        self.assertAlmostEqual(result["variance"][0][1], 0.6666666666666666, places=12)

    def test_axes_sont_des_listes(self):
        result = calculate([0, 1, 2, 3, 4, 5, 6, 7, 8])
        for key in ("mean", "variance", "standard deviation", "max", "min", "sum"):
            self.assertIsInstance(result[key], list)
            self.assertEqual(len(result[key]), 3)
            self.assertIsInstance(result[key][0], list)
            self.assertIsInstance(result[key][1], list)
            self.assertIsInstance(result[key][2], (int, float))


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
