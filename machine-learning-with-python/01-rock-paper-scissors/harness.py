"""Banc de comparaison des stratégies.

Rejoue les matchs sans passer par `play()` de RPS_game.py, pour deux raisons :
`play()` ne rapporte qu'un taux de victoire, alors que je veux l'historique
complet des prédictions ; et il faut isoler l'état des bots entre deux matchs.

Les bots de RPS_game.py accumulent un état dans leurs arguments par défaut
mutables (voir docs/freecodecamp/implementation-notes.md) : `abbey` en particulier
cumule sur *tout* l'historique du process. Comparer la stratégie A puis la
stratégie B dans le même run donnerait un résultat faussé pour B. La factory
ci-dessous recrée un bot neuf à chaque match. C'est l'alternative proportionnée
retenue au cadrage, sans multiprocessing.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import RPS_game
from decision import BEATS, DECISIONS
from metrics import METRICS, wilson_interval
from strategies import STRATEGIES, Distribution, StrategyFn, validate

BOT_NAMES = ("quincy", "abbey", "kris", "mrugesh")


def make_bot(name: str) -> Callable[[str], str]:
    """Recrée un bot avec un état vierge.

    Recharge la fonction depuis le module et lui redonne des arguments par défaut
    neufs, sans toucher à RPS_game.py (interdit par l'énoncé).
    """
    original = getattr(RPS_game, name)

    # Les conteneurs mutables sont VIDÉS, pas copiés : les copier reproduirait
    # l'historique accumulé, ce qui laissait fuir l'état d'un match à l'autre.
    # Les compteurs de `abbey` (dicts imbriqués dans une liste) sont remis à zéro
    # en conservant leurs clés, que le bot indexe sans vérifier leur présence.
    def blank(value):
        if isinstance(value, list):
            # Deux formes distinctes de listes par défaut dans RPS_game.py :
            #   - un historique de coups (`opponent_history=[]`) : à vider ;
            #   - un conteneur de structures (`counter=[0]`, `play_order=[{...}]`)
            #     dont la forme fait partie du contrat : à conserver, remis à zéro.
            if all(isinstance(item, str) for item in value):
                return []
            return [blank(item) for item in value]
        if isinstance(value, dict):
            return {key: blank(sub) for key, sub in value.items()}
        if isinstance(value, int) and not isinstance(value, bool):
            return 0
        return value

    original.__defaults__ = tuple(blank(d) for d in original.__defaults__ or ())

    # `bot` lit les défauts à chaque appel via `original`, donc il voit bien les
    # conteneurs neufs installés ci-dessus.
    def bot(prev_play: str) -> str:
        return original(prev_play)

    return bot


@dataclass
class MatchResult:
    """Résultat détaillé d'un match : bien plus que le taux de victoire."""

    strategy: str
    bot: str
    decision: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    predictions: list[Distribution] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def decided(self) -> int:
        """Parties non nulles. C'est le dénominateur qu'utilise play()."""
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Taux de victoire au sens freeCodeCamp : victoires / parties décidées.

        play() calcule `results['p1'] / (results['p1'] + results['p2']) * 100`,
        les égalités sont exclues du dénominateur. Reproduit ici à l'identique,
        sans quoi le seuil des 60 % ne voudrait pas dire la même chose.
        """
        return 100 * self.wins / self.decided if self.decided else 0.0

    @property
    def win_rate_interval(self) -> tuple[float, float]:
        low, high = wilson_interval(self.wins, self.decided)
        return (100 * low, 100 * high)

    def score(self, metric: str) -> float:
        return METRICS[metric](self.predictions, self.outcomes)


def play_match(
    strategy: StrategyFn,
    bot_name: str,
    games: int = 1000,
    decision: str = "best_response",
    strategy_name: str = "?",
) -> MatchResult:
    """Joue un match complet en conservant chaque prédiction."""
    bot = make_bot(bot_name)
    decide = DECISIONS[decision]
    result = MatchResult(strategy=strategy_name, bot=bot_name, decision=decision)

    opponent_history: list[str] = []
    own_history: list[str] = []
    our_prev_move = ""

    for _ in range(games):
        distribution = strategy(opponent_history, own_history)
        validate(distribution)

        # Les deux coups sont simultanés : le bot ne voit que mon coup du tour
        # précédent, comme dans play(). Lui passer our_move ici lui donnerait un
        # tour d'avance et le rendrait imbattable (kris et abbey me contrent
        # directement).
        our_move = decide(distribution)
        their_move = bot(our_prev_move)
        our_prev_move = our_move

        # La prédiction porte sur le coup adverse : je l'enregistre avant de la vérifier.
        result.predictions.append(dict(distribution))
        result.outcomes.append(their_move)

        if BEATS[their_move] == our_move:
            result.wins += 1
        elif our_move == their_move:
            result.ties += 1
        else:
            result.losses += 1

        opponent_history.append(their_move)
        own_history.append(our_move)

    return result


def compare(
    strategy_names: Sequence[str] = (),
    bot_names: Sequence[str] = BOT_NAMES,
    games: int = 1000,
    decision: str = "best_response",
) -> list[MatchResult]:
    """Croise chaque stratégie avec chaque bot. Sans argument, prend tout le registre."""
    names = strategy_names or tuple(STRATEGIES)
    return [
        play_match(STRATEGIES[name], bot, games, decision, strategy_name=name)
        for name in names
        for bot in bot_names
    ]


def format_table(results: Sequence[MatchResult], metric: str = "brier") -> str:
    """Tableau texte : taux de victoire, IC de Wilson, et une métrique au choix."""
    header = (
        f"{'stratégie':<16} {'bot':<9} {'victoires':>10} {'IC 95%':>16} {metric:>9}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        low, high = result.win_rate_interval
        flag = "" if result.win_rate >= 60 else "  <60%"
        lines.append(
            f"{result.strategy:<16} {result.bot:<9} {result.win_rate:>9.1f}% "
            f"{f'[{low:.1f}, {high:.1f}]':>16} {result.score(metric):>9.4f}{flag}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    results = compare()
    print(format_table(results))
