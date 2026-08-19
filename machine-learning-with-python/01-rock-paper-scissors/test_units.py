"""Tests unitaires des modules du projet.

Distincts de `test_module.py`, qui est le test officiel freeCodeCamp : celui-ci
vérifie le résultat final (60 % contre chaque bot) sans jamais tester une
fonction en particulier. Un bug qui fausse les mesures sans casser le taux de
victoire y passe inaperçu, ce qui est exactement arrivé pendant le développement.

Principe retenu : tester les **invariants qui ont déjà cassé** et ceux dont la
violation serait silencieuse, plutôt que viser une couverture systématique. Sur
un projet de cette taille, une suite courte et ciblée vaut mieux qu'une suite
exhaustive qu'on ne lance pas.

Lancement :
    python -m unittest test_units
    python -m unittest discover      # avec les tests officiels
"""

import unittest

import RPS_game
from decision import BEATS, DECISIONS, best_response
from harness import make_bot, play_match
from metrics import METRICS, brier_score, calibration_bins, wilson_interval
from strategies import (
    _MIXTURE_CACHE,
    MOVES,
    STRATEGIES,
    ContractError,
    context_of,
    dirichlet_posterior_mean,
    markov_dirichlet,
    mixture,
    validate,
)


class TestContratStrategie(unittest.TestCase):
    """Le contrat de sortie : une stratégie renvoie une distribution valide.

    Rien ne l'impose statiquement, le registre étant un dictionnaire de fonctions
    (registre de stratégies). Ces tests sont ce qui tient lieu de vérification.
    """

    def test_validate_accepte_une_distribution_correcte(self):
        validate({"R": 0.5, "P": 0.3, "S": 0.2})

    def test_validate_rejette_une_somme_differente_de_un(self):
        with self.assertRaises(ContractError):
            validate({"R": 0.5, "P": 0.3, "S": 0.9})

    def test_validate_rejette_des_cles_incorrectes(self):
        with self.assertRaises(ContractError):
            validate({"R": 0.5, "P": 0.5, "X": 0.0})

    def test_validate_rejette_une_probabilite_negative(self):
        with self.assertRaises(ContractError):
            validate({"R": 1.2, "P": -0.2, "S": 0.0})

    def test_validate_tolere_l_imprecision_flottante(self):
        """1/3 trois fois ne fait pas exactement 1 en virgule flottante."""
        validate({move: 1 / 3 for move in MOVES})

    def test_toutes_les_strategies_respectent_le_contrat_au_premier_tour(self):
        """Historiques vides : le cas où un comptage brut diviserait par zéro.

        Boucle sur le registre, donc toute stratégie ajoutée plus tard est
        couverte sans modifier ce test.
        """
        for name, strategy in STRATEGIES.items():
            with self.subTest(strategy=name):
                validate(strategy([], []))

    def test_toutes_les_strategies_respectent_le_contrat_en_cours_de_partie(self):
        opponent = ["R", "P", "S", "R", "R", "P", "S", "S", "R", "P"]
        own = ["P", "S", "R", "P", "P", "S", "R", "R", "P", "S"]
        for name, strategy in STRATEGIES.items():
            with self.subTest(strategy=name):
                validate(strategy(opponent, own))


class TestEstimationDirichlet(unittest.TestCase):
    """Moyenne a posteriori d'un Dirichlet-Catégoriel."""

    def test_sans_observation_renvoie_le_prior_uniforme(self):
        from collections import Counter

        distribution = dirichlet_posterior_mean(Counter())
        for move in MOVES:
            self.assertAlmostEqual(distribution[move], 1 / 3)

    def test_lissage_add_one(self):
        """(n_i + 1) / (n + 3) : la formule de Laplace, vérifiée à la main."""
        from collections import Counter

        distribution = dirichlet_posterior_mean(Counter({"R": 2, "P": 1}))
        # n = 3, donc denominateur = 3 + 3 = 6
        self.assertAlmostEqual(distribution["R"], 3 / 6)
        self.assertAlmostEqual(distribution["P"], 2 / 6)
        self.assertAlmostEqual(distribution["S"], 1 / 6)

    def test_le_contexte_d_ordre_zero_est_vide(self):
        self.assertEqual(context_of(["R", "P", "S"], 0), ())

    def test_le_contexte_prend_les_derniers_coups(self):
        self.assertEqual(context_of(["R", "P", "S"], 2), ("P", "S"))

    def test_markov_detecte_une_sequence_deterministe(self):
        """Après 'RR', l'adversaire joue toujours 'S' : la masse doit y aller."""
        history = ["R", "R", "S"] * 20
        distribution = markov_dirichlet(history[:-1] + ["R", "R"], order=2)
        self.assertEqual(max(MOVES, key=lambda m: distribution[m]), "S")


class TestIsolationDesBots(unittest.TestCase):
    """Régression : `make_bot` doit vider l'état, pas le copier.

    Ce test correspond à un bug réel. `make_bot` recopiait les historiques
    accumulés au lieu de les remettre à zéro, si bien qu'abbey arrivait au match
    suivant avec la mémoire du précédent. Le harness affichait alors 78 % là où le
    test officiel donnait 59,7 %, sans qu'aucune erreur ne soit levée.
    """

    def test_abbey_repart_d_un_historique_vide(self):
        bot = make_bot("abbey")
        for _ in range(10):
            bot("R")
        self.assertEqual(len(RPS_game.abbey.__defaults__[0]), 10)

        make_bot("abbey")
        self.assertEqual(len(RPS_game.abbey.__defaults__[0]), 0)

    def test_abbey_repart_d_une_table_de_comptage_vierge(self):
        bot = make_bot("abbey")
        for _ in range(10):
            bot("R")
        make_bot("abbey")
        table = RPS_game.abbey.__defaults__[1][0]
        self.assertTrue(all(count == 0 for count in table.values()))

    def test_quincy_repart_d_un_compteur_a_zero(self):
        bot = make_bot("quincy")
        for _ in range(5):
            bot("")
        make_bot("quincy")
        self.assertEqual(RPS_game.quincy.__defaults__[0], [0])

    def test_deux_matchs_identiques_donnent_le_meme_resultat(self):
        """Sans isolation, le second match serait faussé par le premier."""
        premier = play_match(STRATEGIES["markov_order_2"], "abbey", games=200)
        second = play_match(STRATEGIES["markov_order_2"], "abbey", games=200)
        self.assertEqual(premier.wins, second.wins)


class TestCacheDeLaMixture(unittest.TestCase):
    """`mixture` garde un état de module entre deux appels.

    Même famille de risque que les mutable defaults des bots, et sur la stratégie
    du livrable cette fois : `_MIXTURE_CACHE` accumule les scores des experts d'un
    tour à l'autre, et doit repartir de zéro à chaque nouveau match. La détection
    repose sur un compteur de tours qui recule, mécanisme qu'aucun test ne
    verrouillait jusqu'ici.
    """

    def setUp(self):
        _MIXTURE_CACHE.clear()

    def test_le_cache_est_vide_au_depart(self):
        mixture([], [])
        self.assertEqual(_MIXTURE_CACHE["turn"], 0)
        self.assertTrue(all(not v for v in _MIXTURE_CACHE["scores"].values()))

    def test_le_cache_accumule_pendant_un_match(self):
        opponent, own = [], []
        for coup in "RPSRPSRPSR":
            mixture(opponent, own)
            opponent.append(coup)
            own.append("R")
        # Le dernier appel a eu lieu avec 9 coups déjà joués : `turn` compte les
        # coups vus au moment de l'appel, pas le nombre d'appels.
        self.assertEqual(_MIXTURE_CACHE["turn"], 9)
        self.assertTrue(any(v for v in _MIXTURE_CACHE["scores"].values()))

    def test_un_nouveau_match_reinitialise_le_cache(self):
        """Le compteur de tours qui recule doit déclencher la remise à zéro."""
        opponent, own = [], []
        for coup in "RPSRPSRPSR":
            mixture(opponent, own)
            opponent.append(coup)
            own.append("R")
        self.assertEqual(_MIXTURE_CACHE["turn"], 9)

        mixture([], [])  # nouveau match : historiques vides
        self.assertEqual(_MIXTURE_CACHE["turn"], 0)
        self.assertTrue(all(not v for v in _MIXTURE_CACHE["scores"].values()))

    def test_deux_matchs_consecutifs_donnent_le_meme_resultat(self):
        """Le test qui compte : sans réinitialisation, le second serait faussé."""
        premier = play_match(STRATEGIES["mixture"], "abbey", games=200)
        second = play_match(STRATEGIES["mixture"], "abbey", games=200)
        self.assertEqual(premier.wins, second.wins)
        self.assertEqual(premier.losses, second.losses)

    def test_la_fenetre_borne_l_historique_des_scores(self):
        """Au-delà de `window`, les scores anciens sont oubliés."""
        opponent, own = [], []
        for index in range(150):
            mixture(opponent, own, window=60)
            opponent.append("RPS"[index % 3])
            own.append("R")
        for nom, valeurs in _MIXTURE_CACHE["scores"].items():
            with self.subTest(expert=nom):
                self.assertLessEqual(len(valeurs), 60)


class TestDecision(unittest.TestCase):
    """Transformation d'une distribution en coup joué."""

    def test_beats_est_coherent(self):
        """Chaque coup en bat exactement un autre, sans cycle dégénéré."""
        self.assertEqual(set(BEATS), set(MOVES))
        self.assertEqual(set(BEATS.values()), set(MOVES))
        for move in MOVES:
            self.assertNotEqual(BEATS[move], move)

    def test_best_response_contre_le_coup_le_plus_probable(self):
        distribution = {"R": 0.7, "P": 0.2, "S": 0.1}
        self.assertEqual(best_response(distribution), BEATS["R"])

    def test_toutes_les_decisions_renvoient_un_coup_valide(self):
        distribution = {"R": 0.5, "P": 0.3, "S": 0.2}
        for name, decide in DECISIONS.items():
            with self.subTest(decision=name):
                self.assertIn(decide(distribution), MOVES)


class TestMetriques(unittest.TestCase):
    """Métriques d'évaluation, en particulier leurs cas limites."""

    def test_brier_parfait_vaut_zero(self):
        predictions = [{"R": 1.0, "P": 0.0, "S": 0.0}]
        self.assertAlmostEqual(brier_score(predictions, ["R"]), 0.0)

    def test_brier_maximal_vaut_deux(self):
        """Convention retenue : borne haute à 2, pas à 1."""
        predictions = [{"R": 1.0, "P": 0.0, "S": 0.0}]
        self.assertAlmostEqual(brier_score(predictions, ["P"]), 2.0)

    def test_brier_d_une_prediction_uniforme(self):
        """3 × (1/3)² écarts : valeur de référence d'une croyance neutre."""
        predictions = [{move: 1 / 3 for move in MOVES}]
        self.assertAlmostEqual(brier_score(predictions, ["R"]), 2 / 3)

    def test_les_metriques_gerent_un_historique_vide(self):
        for name, metric in METRICS.items():
            with self.subTest(metric=name):
                self.assertNotEqual(metric([], []), None)

    def test_wilson_ne_divise_pas_par_zero(self):
        low, high = wilson_interval(0, 0)
        self.assertNotEqual(low, low)  # NaN, seul objet différent de lui-même

    def test_wilson_reste_dans_zero_un(self):
        for successes, trials in [(0, 10), (10, 10), (5, 10), (1, 3)]:
            with self.subTest(successes=successes, trials=trials):
                low, high = wilson_interval(successes, trials)
                self.assertGreaterEqual(low, 0.0)
                self.assertLessEqual(high, 1.0)
                self.assertLessEqual(low, high)

    def test_wilson_encadre_la_proportion(self):
        low, high = wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_calibration_ignore_les_intervalles_vides(self):
        predictions = [{"R": 1.0, "P": 0.0, "S": 0.0}] * 5
        bins = calibration_bins(predictions, ["R"] * 5)
        self.assertTrue(all(effectif > 0 for _, _, effectif in bins))


class TestHarness(unittest.TestCase):
    """Comptage des résultats d'un match."""

    def test_les_comptes_somment_au_nombre_de_parties(self):
        result = play_match(STRATEGIES["uniform"], "kris", games=100)
        self.assertEqual(result.wins + result.losses + result.ties, 100)

    def test_le_taux_exclut_les_egalites_comme_freecodecamp(self):
        """play() divise par les parties décidées, pas par le total."""
        result = play_match(STRATEGIES["markov_order_2"], "quincy", games=200)
        attendu = 100 * result.wins / (result.wins + result.losses)
        self.assertAlmostEqual(result.win_rate, attendu)

    def test_une_prediction_est_conservee_par_tour(self):
        result = play_match(STRATEGIES["markov_order_1"], "kris", games=50)
        self.assertEqual(len(result.predictions), 50)
        self.assertEqual(len(result.outcomes), 50)


if __name__ == "__main__":
    unittest.main()
