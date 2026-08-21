"""Ratings for the arena.

Two estimators, on purpose:

* **Online ELO** -- cheap, updates per vote, drives the live leaderboard.
  It is *order-dependent*: the same votes in a different sequence give different
  numbers, which matters at the vote counts here (100-2000).
* **Bradley-Terry MLE** -- order-independent maximum likelihood, with bootstrap
  confidence intervals. This is what the report leads with, so it can say "these
  two are indistinguishable" instead of implying a ranking that the data does
  not support.

Draws (including `both_bad` and auto-draws on byte-identical output) count as
half a win to each side -- the standard Rao-Kupper simplification.

No numpy/scipy: the MM iteration below is a dozen lines and keeps this runnable
in any of the three interpreters this project uses.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field


DEFAULT_INITIAL = 1500.0
DEFAULT_K = 24.0
SCALE = 400.0
# Virtual games per entrant used to regularize the Bradley-Terry fit; see
# bradley_terry() for why an unregularized MLE is unusable early in a session.
PRIOR_GAMES = 2.0


@dataclass
class Record:
    wins: float = 0.0
    losses: float = 0.0
    draws: int = 0
    games: int = 0


@dataclass
class Ratings:
    elo: dict[str, float] = field(default_factory=dict)
    bt: dict[str, float] = field(default_factory=dict)
    ci_low: dict[str, float] = field(default_factory=dict)
    ci_high: dict[str, float] = field(default_factory=dict)
    records: dict[str, Record] = field(default_factory=dict)
    pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the ELO/logistic model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))


def _outcome(winner: str) -> float:
    """Score for entrant A: 1 win, 0 loss, 0.5 draw."""
    if winner == "a":
        return 1.0
    if winner == "b":
        return 0.0
    return 0.5  # draw / both_bad / auto


def online_elo(
    votes: list[dict],
    entrants: list[str],
    initial: float = DEFAULT_INITIAL,
    k_factor: float = DEFAULT_K,
) -> tuple[dict[str, float], dict[str, Record], dict[tuple[str, str], int]]:
    elo = {name: initial for name in entrants}
    records = {name: Record() for name in entrants}
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for vote in votes:
        a, b = vote["entrant_a"], vote["entrant_b"]
        if a not in elo or b not in elo:
            continue  # entrant removed from the registry since the vote
        score = _outcome(vote["winner"])
        expected = expected_score(elo[a], elo[b])
        elo[a] += k_factor * (score - expected)
        elo[b] += k_factor * ((1.0 - score) - (1.0 - expected))

        pair_counts[_pair_key(a, b)] += 1
        for name in (a, b):
            records[name].games += 1
        if score == 1.0:
            records[a].wins += 1
            records[b].losses += 1
        elif score == 0.0:
            records[b].wins += 1
            records[a].losses += 1
        else:
            records[a].draws += 1
            records[b].draws += 1

    return elo, records, dict(pair_counts)


def bradley_terry(
    votes: list[dict],
    entrants: list[str],
    iterations: int = 400,
    tolerance: float = 1e-9,
    prior_games: float = PRIOR_GAMES,
) -> dict[str, float]:
    """Bradley-Terry strengths by MM iteration, returned on the ELO scale.

    Solves p_i = W_i / sum_j (n_ij / (p_i + p_j)), the standard minorize-maximize
    update. Order-independent, unlike sequential ELO.

    `prior_games` adds a symmetric shrinkage prior: each entrant gets that many
    virtual games, split half-won and half-lost, against a phantom opponent of
    fixed average strength. Without it an undefeated entrant has an unbounded
    MLE -- a single 1-0 record produced a rating of 2817 in testing, which makes
    the live leaderboard nonsense in the first minutes of a session. With it,
    ratings start pinned near the mean and separate only as evidence accumulates.
    Standard practice; the effect is negligible once a pair has real games.
    """
    wins: dict[str, float] = {name: 0.0 for name in entrants}
    pair_games: dict[tuple[str, str], float] = defaultdict(float)
    anchor_strength = 1.0

    for vote in votes:
        a, b = vote["entrant_a"], vote["entrant_b"]
        if a not in wins or b not in wins:
            continue
        score = _outcome(vote["winner"])
        wins[a] += score
        wins[b] += 1.0 - score
        pair_games[_pair_key(a, b)] += 1.0

    active = [name for name in entrants if any(name in key for key in pair_games)]
    if not active:
        return {name: DEFAULT_INITIAL for name in entrants}

    # Shrinkage prior: half a win and half a loss per virtual game, against a
    # phantom opponent of fixed strength 1.0.
    for name in active:
        wins[name] += prior_games / 2.0

    strength = {name: 1.0 for name in active}
    neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), count in pair_games.items():
        neighbours[a].append((b, count))
        neighbours[b].append((a, count))

    for _ in range(iterations):
        largest_change = 0.0
        for name in active:
            denominator = sum(
                count / (strength[name] + strength[other])
                for other, count in neighbours[name]
                if other in strength
            )
            denominator += prior_games / (strength[name] + anchor_strength)
            new_value = wins[name] / denominator if denominator > 0 else strength[name]
            largest_change = max(largest_change, abs(new_value - strength[name]))
            strength[name] = new_value

        geometric_mean = math.exp(
            sum(math.log(max(value, 1e-12)) for value in strength.values())
            / len(strength)
        )
        strength = {name: value / geometric_mean for name, value in strength.items()}
        if largest_change < tolerance:
            break

    # ln(strength) * 400/ln(10), anchored so the mean sits at the initial rating.
    factor = SCALE / math.log(10.0)
    logits = {
        name: math.log(max(value, 1e-12)) * factor for name, value in strength.items()
    }
    offset = DEFAULT_INITIAL - (sum(logits.values()) / len(logits))
    rated = {name: value + offset for name, value in logits.items()}
    for name in entrants:
        rated.setdefault(name, DEFAULT_INITIAL)
    return rated


def bootstrap_intervals(
    votes: list[dict],
    entrants: list[str],
    rounds: int = 200,
    seed: int = 20260818,
) -> tuple[dict[str, float], dict[str, float]]:
    """Percentile bootstrap over votes, resampled with replacement."""
    if len(votes) < 10:
        return ({}, {})

    generator = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    count = len(votes)
    for _ in range(rounds):
        resampled = [votes[generator.randrange(count)] for _ in range(count)]
        fitted = bradley_terry(resampled, entrants, iterations=120)
        for name, value in fitted.items():
            samples[name].append(value)

    low: dict[str, float] = {}
    high: dict[str, float] = {}
    for name, values in samples.items():
        values.sort()
        low[name] = values[int(0.025 * len(values))]
        high[name] = values[min(int(0.975 * len(values)), len(values) - 1)]
    return low, high


def compute(
    votes: list[dict],
    entrants: list[str],
    initial: float = DEFAULT_INITIAL,
    k_factor: float = DEFAULT_K,
    bootstrap_rounds: int = 0,
) -> Ratings:
    elo, records, pair_counts = online_elo(votes, entrants, initial, k_factor)
    bt = bradley_terry(votes, entrants)
    low, high = ({}, {})
    if bootstrap_rounds:
        low, high = bootstrap_intervals(votes, entrants, bootstrap_rounds)
    return Ratings(
        elo=elo,
        bt=bt,
        ci_low=low,
        ci_high=high,
        records=records,
        pair_counts=pair_counts,
    )


def win_matrix(votes: list[dict], entrants: list[str]) -> dict[str, dict[str, dict]]:
    """Head-to-head tallies: matrix[a][b] = {'wins','losses','draws','games'}."""
    matrix = {
        a: {
            b: {"wins": 0.0, "losses": 0.0, "draws": 0, "games": 0}
            for b in entrants
            if b != a
        }
        for a in entrants
    }
    for vote in votes:
        a, b = vote["entrant_a"], vote["entrant_b"]
        if a not in matrix or b not in matrix.get(a, {}):
            continue
        score = _outcome(vote["winner"])
        matrix[a][b]["games"] += 1
        matrix[b][a]["games"] += 1
        if score == 1.0:
            matrix[a][b]["wins"] += 1
            matrix[b][a]["losses"] += 1
        elif score == 0.0:
            matrix[a][b]["losses"] += 1
            matrix[b][a]["wins"] += 1
        else:
            matrix[a][b]["draws"] += 1
            matrix[b][a]["draws"] += 1
    return matrix


def position_bias(votes: list[dict]) -> dict[str, float]:
    """Did the left column win more often? A check on reviewer consistency.

    entrant_a/entrant_b are stored as displayed (left/right), so this stays
    measurable after the fact.
    """
    human = [vote for vote in votes if not vote.get("auto")]
    decided = [vote for vote in human if vote["winner"] in ("a", "b")]
    if not decided:
        return {"left_win_rate": 0.0, "decided": 0, "draws": len(human) - len(decided)}
    left = sum(1 for vote in decided if vote["winner"] == "a")
    return {
        "left_win_rate": left / len(decided),
        "decided": len(decided),
        "draws": len(human) - len(decided),
    }


def position_bias_controlled(votes: list[dict]) -> dict:
    """First-slot advantage, controlled for which entrants sat in which slot.

    The raw left-win rate confounds two things. If the stronger entrant happens to
    be shown first more often, the left column wins more for reasons that have
    nothing to do with position. Comparing against Bradley-Terry ratings does not
    settle it either, because those ratings are fitted from these same votes: a
    genuine left-side advantage inflates whichever entrants sat left, so the
    comparison is circular.

    This restricts to unordered pairs that have votes in *both* orders and asks a
    question with no strength term in it: within a pair, does the same entrant win
    more often when shown first than when shown second? Both slots hold the same
    two entrants, so any difference is positional.
    """
    from collections import defaultdict

    decided = [
        vote for vote in votes if not vote.get("auto") and vote["winner"] in ("a", "b")
    ]
    by_pair: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for vote in decided:
        key = _pair_key(vote["entrant_a"], vote["entrant_b"])
        slot = by_pair[key][vote["entrant_a"]]
        slot[0] += vote["winner"] == "a"
        slot[1] += 1

    first_wins = first_games = second_wins = second_games = 0
    pairs = 0
    for (left, right), slots in by_pair.items():
        if len(slots) < 2:
            continue  # only one ordering seen: nothing to control against
        shown_first = slots[left]
        # `left` shown second == the games where `right` was first and lost.
        opposite = slots[right]
        shown_second = [opposite[1] - opposite[0], opposite[1]]
        if not shown_first[1] or not shown_second[1]:
            continue
        pairs += 1
        first_wins += shown_first[0]
        first_games += shown_first[1]
        second_wins += shown_second[0]
        second_games += shown_second[1]

    if not first_games or not second_games:
        return {"pairs": 0, "advantage": None}
    first_rate = first_wins / first_games
    second_rate = second_wins / second_games
    return {
        "pairs": pairs,
        "first_rate": first_rate,
        "second_rate": second_rate,
        "first_games": first_games,
        "second_games": second_games,
        "advantage": first_rate - second_rate,
    }
