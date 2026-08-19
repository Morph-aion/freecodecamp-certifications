# Livrable freeCodeCamp. Couche mince d'adaptation : toute la logique vit dans
# strategies.py et decision.py (convention du projet 01), il n'y a aucune duplication.
#
# Stratégie retenue : mixture d'experts. Aucun expert unique ne bat les 4 bots :
# les chaînes de Markov démasquent quincy/kris/mrugesh mais plafonnent à ~60%
# contre abbey, qui modélise mon propre jeu ; self_model fait l'inverse. La
# sélection par score de Brier sur fenêtre glissante tranche à l'exécution.
# Comparaison complète des experts : harness.py.

from decision import best_response
from strategies import mixture

_opponent_history: list[str] = []
_own_history: list[str] = []


def player(prev_play, opponent_history=[]):
    """Renvoie le prochain coup à jouer ("R", "P" ou "S").

    `prev_play` est le dernier coup de l'adversaire, chaîne vide au premier tour
    du match. Le second argument fait partie de la signature imposée par
    freeCodeCamp mais n'est pas utilisé : l'état est géré par les listes de module
    ci-dessus, réinitialisées à chaque nouveau match.
    """
    if not prev_play:
        # Chaîne vide = premier tour d'un nouveau match : on repart d'un état vierge,
        # sans quoi l'historique du match précédent contaminerait celui-ci.
        _opponent_history.clear()
        _own_history.clear()
    else:
        _opponent_history.append(prev_play)

    distribution = mixture(_opponent_history, _own_history)
    move = best_response(distribution)
    _own_history.append(move)
    return move
