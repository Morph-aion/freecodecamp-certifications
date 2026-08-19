"""Registre de métriques d'évaluation (registre progressif).

Chaque métrique prend l'historique des prédictions et les issues observées, et
renvoie un scalaire. Le contrat de sortie (une stratégie renvoie une distribution)
rend calculables dès la première stratégie les scores probabilistes, qui
dégénéreraient sur une prédiction ponctuelle.

Convention de Brier retenue : BS = Σᵢ(pᵢ − δᵢⱼ)², bornée [0, 2]. D'autres
conventions divisent par 2 ou par K, d'où l'idée répandue que « le Brier est
entre 0 et 1 ». À garder en tête pour comparer avec des chiffres publiés.
"""

import math
from collections.abc import Callable, Sequence

from strategies import MOVES, Distribution

# Probabilité plancher pour le log-loss : sans elle, une prédiction certaine et
# fausse renverrait l'infini. Le score dépend alors de ce choix : le signaler
# plutôt que de le cacher.
LOG_LOSS_FLOOR = 1e-15

MetricFn = Callable[[Sequence[Distribution], Sequence[str]], float]

METRICS: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if name in METRICS:
            raise ValueError(f"métrique {name!r} déjà enregistrée")
        METRICS[name] = fn
        return fn

    return decorator


@register("brier")
def brier_score(predictions: Sequence[Distribution], outcomes: Sequence[str]) -> float:
    """Score de Brier multi-catégoriel moyen. Plus bas vaut mieux.

    Règle de score strictement propre : optimisée en espérance uniquement si l'on
    rapporte sa croyance honnête (Gneiting & Raftery 2007).
    """
    if not predictions:
        return float("nan")
    total = 0.0
    for distribution, actual in zip(predictions, outcomes, strict=True):
        total += sum((distribution[move] - (move == actual)) ** 2 for move in MOVES)
    return total / len(predictions)


@register("log_loss")
def log_loss(predictions: Sequence[Distribution], outcomes: Sequence[str]) -> float:
    """Log-loss moyen (score logarithmique). Plus bas vaut mieux.

    Seule règle propre *locale* : ne dépend que de la probabilité attribuée à
    l'issue réalisée.
    """
    if not predictions:
        return float("nan")
    total = 0.0
    for distribution, actual in zip(predictions, outcomes, strict=True):
        total -= math.log(max(distribution[actual], LOG_LOSS_FLOOR))
    return total / len(predictions)


@register("accuracy")
def accuracy(predictions: Sequence[Distribution], outcomes: Sequence[str]) -> float:
    """Taux de prédiction correcte de l'argmax. Plus haut vaut mieux.

    Volontairement conservée à côté du Brier : leur écart est précisément ce que
    je perdrais à ne renvoyer qu'un coup au lieu d'une distribution.
    """
    if not predictions:
        return float("nan")
    correct = sum(
        max(MOVES, key=lambda move: distribution[move]) == actual
        for distribution, actual in zip(predictions, outcomes, strict=True)
    )
    return correct / len(predictions)


def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion.

    Préféré à l'intervalle de Wald, qui se comporte mal près de 0 et 1 et sur
    petits effectifs, précisément le régime des contextes rarement visités.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    )
    margin /= denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def calibration_bins(
    predictions: Sequence[Distribution],
    outcomes: Sequence[str],
    bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Courbe de calibration : (confiance moyenne, fréquence observée, effectif).

    Un modèle calibré annonce 70 % sur des événements qui se produisent 70 % du
    temps. C'est ce diagnostic qu'une prédiction ponctuelle rend impossible.
    """
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for distribution, actual in zip(predictions, outcomes, strict=True):
        for move in MOVES:
            probability = distribution[move]
            index = min(int(probability * bins), bins - 1)
            buckets[index].append((probability, move == actual))

    result = []
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(hit for _, hit in bucket) / len(bucket)
        result.append((mean_confidence, observed, len(bucket)))
    return result
