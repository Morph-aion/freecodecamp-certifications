"""Registre de stratégies de prédiction interchangeables.

Une stratégie prédit le *prochain* coup de l'adversaire sous forme de distribution
de probabilité sur {"R", "P", "S"}, jamais un coup unique.

Contrat (contrat de sortie probabiliste) :
    (opponent_history, own_history) -> {"R": float, "P": float, "S": float}

Les deux historiques sont chronologiques. Une stratégie qui n'a pas besoin de
`own_history` l'ignore simplement : la signature reste uniforme, condition de
l'interchangeabilité.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

MOVES = ("R", "P", "S")

Distribution = Mapping[str, float]
StrategyFn = Callable[[Sequence[str], Sequence[str]], Distribution]

STRATEGIES: dict[str, StrategyFn] = {}

# Prior de Laplace. Dir(1,1,1) est le prior plat sur le simplexe, dont la moyenne
# a posteriori est exactement le lissage add-one. Voir markov-multinomial-bayesien.md.
DEFAULT_ALPHA = 1.0


class ContractError(ValueError):
    """Levée quand une stratégie renvoie autre chose qu'une distribution valide."""


def validate(distribution: Distribution, *, tolerance: float = 1e-9) -> None:
    """Vérifie le contrat de sortie à l'exécution.

    Le registre utilise délibérément des fonctions simples plutôt qu'une ABC
    (registre de stratégies) : rien n'impose donc le contrat statiquement, et cette fonction en
    tient lieu.

    Appelée à chaque tour par `harness.play_match`, et sur toutes les stratégies
    du registre par `test_units.TestContratStrategie`. Elle n'est délibérément
    **pas** appelée par `register` : une stratégie ne produit une distribution
    qu'à l'exécution, il n'y a rien à valider au moment où on l'enregistre.
    """
    if set(distribution) != set(MOVES):
        raise ContractError(f"clés attendues {set(MOVES)}, reçu {set(distribution)}")
    if any(p < 0 for p in distribution.values()):
        raise ContractError(f"probabilités négatives : {dict(distribution)}")
    total = sum(distribution.values())
    if abs(total - 1.0) > tolerance:
        raise ContractError(f"la somme doit valoir 1, reçu {total}")


def register(name: str) -> Callable[[StrategyFn], StrategyFn]:
    """Ajoute une stratégie au registre sous le nom `name`."""

    def decorator(fn: StrategyFn) -> StrategyFn:
        if name in STRATEGIES:
            raise ValueError(f"stratégie {name!r} déjà enregistrée")
        STRATEGIES[name] = fn
        return fn

    return decorator


def dirichlet_posterior_mean(
    counts: Counter, alpha: float = DEFAULT_ALPHA
) -> dict[str, float]:
    """Moyenne a posteriori d'un modèle Dirichlet-Catégoriel : (n_i + α) / (n + K·α).

    Avec α = 1, c'est le lissage add-one. Ne divise jamais par zéro : sans aucune
    observation, renvoie la moyenne du prior, ce qui fait fonctionner le tout
    premier tour sans cas particulier.
    """
    total = sum(counts[move] for move in MOVES)
    denominator = total + len(MOVES) * alpha
    return {move: (counts[move] + alpha) / denominator for move in MOVES}


def context_of(history: Sequence[str], order: int) -> tuple[str, ...]:
    """Les `order` derniers coups, qui servent de contexte de conditionnement."""
    if order == 0:
        return ()
    return tuple(history[-order:])


def markov_dirichlet(
    opponent_history: Sequence[str],
    order: int,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Chaîne de Markov d'ordre k à émissions catégorielles, estimée par contexte.

    Un Dirichlet par contexte. Un contexte jamais visité retombe sur le prior :
    c'est précisément l'intérêt d'une moyenne a posteriori plutôt que de
    fréquences brutes.
    """
    counts: Counter = Counter()
    current = context_of(opponent_history, order)

    # Parcourt chaque position où un contexte complet existe, et note ce qui a suivi.
    for i in range(order, len(opponent_history)):
        if tuple(opponent_history[i - order : i]) == current:
            counts[opponent_history[i]] += 1

    return dirichlet_posterior_mean(counts, alpha)


def _make_markov_strategy(order: int) -> StrategyFn:
    def strategy(
        opponent_history: Sequence[str], own_history: Sequence[str]
    ) -> dict[str, float]:
        return markov_dirichlet(opponent_history, order)

    strategy.__name__ = f"markov_order_{order}"
    strategy.__doc__ = (
        f"Chaîne de Markov d'ordre {order}, Dirichlet({DEFAULT_ALPHA}) par contexte."
    )
    return strategy


# Étape 1 du chemin d'implémentation : l'ordre 0 est le plancher de référence,
# une seule catégorielle sur les fréquences globales de l'adversaire, sans mémoire.
# Les ordres 1 à 3 répondent empiriquement à la question de l'ordre optimal (étape 2).
for _order in (0, 1, 2, 3):
    register(f"markov_order_{_order}")(_make_markov_strategy(_order))


@register("uniform")
def uniform(
    opponent_history: Sequence[str], own_history: Sequence[str]
) -> dict[str, float]:
    """Témoin : aucune croyance. Une stratégie qui ne le bat pas est cassée."""
    return {move: 1 / len(MOVES) for move in MOVES}


# Non enregistrée telle quelle : son paramètre `order` change tout (l'ordre 1 bat
# abbey à 83 %, l'ordre 2 tombe à 49 %), et l'exposer sous un nom unique masquait
# cette différence. Le registre ne contient que les variantes explicites
# `self_model_1` et `self_model_2`, construites plus bas.
def self_model(
    opponent_history: Sequence[str],
    own_history: Sequence[str],
    order: int = 2,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Modélise *nos propres* coups pour anticiper un adversaire qui nous modélise.

    Contre un adversaire réactif, prédire ses coups directement est une impasse :
    ce qu'il joue est une fonction de ce qu'il croit que je vais jouer. Je modélise
    donc ma propre séquence, celle qu'il observe, puis j'en déduis ce qu'il
    va jouer contre elle.

    C'est le premier niveau méta au sens d'Iocaine Powder (cf. problem-framing.md) :
    prédire non pas l'adversaire, mais l'adversaire en train de me prédire.
    """
    counts: Counter = Counter()
    current = context_of(own_history, order)

    for i in range(order, len(own_history)):
        if tuple(own_history[i - order : i]) == current:
            counts[own_history[i]] += 1

    our_next = dirichlet_posterior_mean(counts, alpha)

    # Il joue ce qui bat ce que je vais jouer : la masse se reporte sur le contre.
    counters = {"R": "P", "P": "S", "S": "R"}
    return {counters[move]: probability for move, probability in our_next.items()}


def _make_self_model_strategy(order: int) -> StrategyFn:
    def strategy(
        opponent_history: Sequence[str], own_history: Sequence[str]
    ) -> dict[str, float]:
        return self_model(opponent_history, own_history, order=order)

    strategy.__name__ = f"self_model_{order}"
    strategy.__doc__ = f"Modèle de mes propres coups à l'ordre {order}."
    return strategy


# L'ordre compte beaucoup ici : seul l'ordre 1 bat abbey (83 %), les ordres
# supérieurs retombent à ~50 %. Les exposer séparément évite de croire que
# `self_model` est bon ou mauvais en bloc.
for _self_order in (1, 2):
    register(f"self_model_{_self_order}")(_make_self_model_strategy(_self_order))


def _brier_of(distribution: Distribution, actual: str) -> float:
    return sum((distribution[move] - (move == actual)) ** 2 for move in MOVES)


@register("mixture")
def mixture(
    opponent_history: Sequence[str],
    own_history: Sequence[str],
    window: int = 60,
) -> dict[str, float]:
    """Sélectionne l'expert le plus performant sur une fenêtre glissante récente.

    Aucun expert ne gagne partout : les chaînes de Markov démasquent les bots à
    séquence fixe, `self_model` bat les bots réactifs, et l'inverse est vrai. Plutôt
    que d'élire un modèle a priori, je note chaque expert au score de Brier sur ses
    prédictions récentes et je suis le meilleur.

    La fenêtre glissante joue le rôle du facteur d'oubli discuté dans
    markov-multinomial-bayesien.md : elle rend la sélection capable de suivre un
    adversaire qui change de comportement en cours de partie.

    Nature exacte de cette règle : c'est un *Follow-the-Leader fenêtré*, au sens
    de la littérature « prediction with expert advice » (Cesa-Bianchi & Lugosi).
    C'en est la variante théoriquement la plus fragile : FTL se comporte bien en
    régime stochastique mais mal face à un adversaire adverse, précisément le cas
    d'abbey. Le remplacement mieux fondé est Hedge (poids exponentiels), qui offre
    une borne de regret : piste restée ouverte.
    """
    experts: dict[str, Callable[[Sequence[str], Sequence[str]], dict[str, float]]] = {
        "markov_2": lambda opp, own: markov_dirichlet(opp, 2),
        "markov_3": lambda opp, own: markov_dirichlet(opp, 3),
        "self_1": lambda opp, own: self_model(opp, own, order=1),
        "self_2": lambda opp, own: self_model(opp, own, order=2),
    }

    # Les scores sont accumulés d'un appel à l'autre plutôt que recalculés sur toute
    # la fenêtre : le cache retient, pour chaque expert, la prédiction qu'il avait
    # faite au tour précédent, que je note dès que l'issue réelle est connue. Sans
    # cela le coût serait quadratique en la longueur du match.
    turn = len(opponent_history)
    cache = _MIXTURE_CACHE
    if not cache or cache["turn"] > turn or turn == 0:
        cache.clear()
        cache.update(turn=0, scores={name: [] for name in experts}, pending=None)

    if cache["pending"] is not None and turn > cache["turn"]:
        actual = opponent_history[-1]
        for name, distribution in cache["pending"].items():
            history = cache["scores"][name]
            history.append(_brier_of(distribution, actual))
            if len(history) > window:
                del history[0]

    predictions = {
        name: fn(opponent_history, own_history) for name, fn in experts.items()
    }
    cache.update(turn=turn, pending=predictions)

    # Tant qu'aucun expert n'a été noté, je retombe sur markov_2 par défaut.
    scored = {name: sum(values) for name, values in cache["scores"].items() if values}
    best = min(scored, key=scored.get) if scored else "markov_2"
    return predictions[best]


# État de la mixture entre deux appels. Réinitialisé dès qu'un nouveau match
# commence (détecté par un compteur de tours qui repart en arrière).
_MIXTURE_CACHE: dict = {}
