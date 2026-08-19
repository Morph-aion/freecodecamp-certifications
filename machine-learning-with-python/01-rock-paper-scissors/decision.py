"""Règles de décision : distribution prédite -> coup à jouer.

Séparé des stratégies (contrat de sortie probabiliste) pour que les deux axes soient expérimentables
indépendamment : même stratégie, plusieurs règles de décision, mesurables
séparément.
"""

import random
from collections.abc import Callable

from strategies import MOVES, Distribution

# Ce qui bat quoi : BEATS[x] est le coup qui l'emporte sur x.
BEATS = {"R": "P", "P": "S", "S": "R"}

DecisionFn = Callable[[Distribution], str]

DECISIONS: dict[str, DecisionFn] = {}


def register(name: str) -> Callable[[DecisionFn], DecisionFn]:
    def decorator(fn: DecisionFn) -> DecisionFn:
        if name in DECISIONS:
            raise ValueError(f"règle de décision {name!r} déjà enregistrée")
        DECISIONS[name] = fn
        return fn

    return decorator


@register("best_response")
def best_response(distribution: Distribution) -> str:
    """Joue ce qui bat le coup adverse le plus probable.

    Optimal en espérance dès lors que la matrice de gains est symétrique, ce qui
    est le cas ici : chaque victoire vaut autant, quel que soit le coup joué.
    Départage déterministe par l'ordre de MOVES pour rester reproductible.
    """
    predicted = max(MOVES, key=lambda move: distribution[move])
    return BEATS[predicted]


@register("sample")
def sample(distribution: Distribution) -> str:
    """Échantillonne le coup adverse selon la distribution, puis joue ce qui le bat.

    Moins performant que best_response contre un adversaire fixe, mais moins
    exploitable par un adversaire qui nous modélise en retour.
    """
    predicted = random.choices(MOVES, weights=[distribution[m] for m in MOVES])[0]
    return BEATS[predicted]
