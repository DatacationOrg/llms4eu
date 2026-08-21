"""Tests for the scraper arena's rating maths."""

import random

from research.scrapers import rating, stats


def vote(a: str, b: str, winner: str, auto: int = 0) -> dict:
    return {"entrant_a": a, "entrant_b": b, "winner": winner, "auto": auto}


def test_elo_hand_computed_win():
    elo, _, _ = rating.online_elo([vote("A", "B", "a")], ["A", "B"], 1500.0, 24.0)
    assert elo["A"] == 1512.0
    assert elo["B"] == 1488.0


def test_elo_draw_between_equals_is_a_no_op():
    elo, _, _ = rating.online_elo([vote("A", "B", "draw")], ["A", "B"], 1500.0, 24.0)
    assert elo["A"] == 1500.0
    assert elo["B"] == 1500.0


def test_expected_score_scale():
    assert rating.expected_score(1500, 1500) == 0.5
    assert abs(rating.expected_score(1900, 1500) - 0.9090909) < 1e-6


def _synthetic_votes(seed: int = 7, count: int = 4000):
    strengths = {"strong": 1900.0, "mid": 1500.0, "weak": 1100.0}
    names = list(strengths)
    generator = random.Random(seed)
    votes = []
    for _ in range(count):
        a, b = generator.sample(names, 2)
        probability = rating.expected_score(strengths[a], strengths[b])
        votes.append(vote(a, b, "a" if generator.random() < probability else "b"))
    return votes, names


def test_bradley_terry_recovers_known_ranking_and_spacing():
    votes, names = _synthetic_votes()
    bt = rating.bradley_terry(votes, names)
    assert bt["strong"] > bt["mid"] > bt["weak"]
    # true spread is 800 ELO points
    assert abs((bt["strong"] - bt["weak"]) - 800) < 80


def test_bradley_terry_is_order_independent_but_elo_is_not():
    votes, names = _synthetic_votes()
    shuffled = votes[:]
    random.Random(11).shuffle(shuffled)

    bt_a = rating.bradley_terry(votes, names)
    bt_b = rating.bradley_terry(shuffled, names)
    assert all(abs(bt_a[name] - bt_b[name]) < 1e-6 for name in names)

    elo_a, _, _ = rating.online_elo(votes, names)
    elo_b, _, _ = rating.online_elo(shuffled, names)
    assert any(abs(elo_a[name] - elo_b[name]) > 1.0 for name in names)


def test_bradley_terry_survives_a_zero_win_entrant():
    bt = rating.bradley_terry([vote("A", "B", "a")] * 5, ["A", "B"])
    assert bt["A"] > bt["B"]
    assert all(abs(value) < 1e6 for value in bt.values())


def test_bootstrap_intervals_bracket_the_point_estimate():
    votes, names = _synthetic_votes(count=1200)
    bt = rating.bradley_terry(votes, names)
    low, high = rating.bootstrap_intervals(votes, names, rounds=40)
    for name in names:
        assert low[name] <= bt[name] <= high[name] + 1e-6


def test_position_bias_counts_only_decided_votes():
    bias = rating.position_bias(
        [vote("A", "B", "a"), vote("A", "B", "b"), vote("A", "B", "draw")]
    )
    assert bias["decided"] == 2
    assert bias["left_win_rate"] == 0.5
    assert bias["draws"] == 1


def test_win_matrix_is_symmetric():
    matrix = rating.win_matrix([vote("A", "B", "a"), vote("A", "B", "b")], ["A", "B"])
    assert matrix["A"]["B"] == {"wins": 1.0, "losses": 1.0, "draws": 0, "games": 2}
    assert matrix["B"]["A"] == {"wins": 1.0, "losses": 1.0, "draws": 0, "games": 2}


def test_shrinkage_keeps_sparse_ratings_sane():
    """A single 1-0 record must not produce a four-digit blowout.

    An unregularized Bradley-Terry MLE is unbounded for an undefeated entrant,
    which made the live leaderboard show 2817 after one vote.
    """
    bt = rating.bradley_terry([vote("A", "B", "a")], ["A", "B"])
    assert bt["A"] > bt["B"]
    assert abs(bt["A"] - rating.DEFAULT_INITIAL) < 200
    assert abs(bt["B"] - rating.DEFAULT_INITIAL) < 200


def test_shrinkage_yields_to_evidence():
    """With plenty of votes the prior stops mattering and spacing is recovered."""
    votes, names = _synthetic_votes(count=4000)
    strong = rating.bradley_terry(votes, names)
    weak_prior = rating.bradley_terry(votes, names, prior_games=0.02)
    assert all(abs(strong[name] - weak_prior[name]) < 25 for name in names)


def test_position_bias_controlled_ignores_strength_composition():
    """A strong entrant always shown first must not read as position bias.

    This is the confound the raw left-win rate cannot separate: `strong` wins every
    game and is always in the left slot, so the raw rate is 100% left. With only one
    ordering per pair there is nothing to control against, so the controlled measure
    must decline to report an advantage rather than invent one.
    """
    votes = [vote("strong", "weak", "a") for _ in range(20)]
    assert rating.position_bias(votes)["left_win_rate"] == 1.0
    assert rating.position_bias_controlled(votes)["advantage"] is None


def test_position_bias_controlled_detects_a_real_first_slot_advantage():
    """Evenly matched pair, both orderings: whoever is shown first always wins."""
    votes = [vote("a1", "b1", "a") for _ in range(10)]
    votes += [vote("b1", "a1", "a") for _ in range(10)]

    # Raw rate cannot tell this apart from "a1 is better".
    assert rating.position_bias(votes)["left_win_rate"] == 1.0
    controlled = rating.position_bias_controlled(votes)
    assert controlled["pairs"] == 1
    assert controlled["advantage"] == 1.0  # wins 100% first, 0% second


def test_position_bias_controlled_reports_no_advantage_when_there_is_none():
    """Same entrant wins at the same rate in both slots."""
    votes = [vote("a1", "b1", "a"), vote("a1", "b1", "b")]
    votes += [vote("b1", "a1", "a"), vote("b1", "a1", "b")]
    controlled = rating.position_bias_controlled(votes)
    assert controlled["advantage"] == 0.0


# ------------------------------------------------------------------- kappa


def test_kappa_is_zero_when_agreement_is_only_chance():
    """Two raters with the same marginals and no real correlation score ~0.

    Both raters pick "A" three quarters of the time; arranged so their agreement is
    exactly the 0.75*0.75 + 0.25*0.25 = 62.5% that chance alone gives.
    """
    labels = [("A", "A")] * 9 + [("A", "B")] * 3 + [("B", "A")] * 3 + [("B", "B")] * 1
    assert stats.cohens_kappa(labels) == 0.0


def test_kappa_is_one_for_perfect_agreement_across_two_labels():
    labels = [("A", "A")] * 5 + [("B", "B")] * 5
    assert stats.cohens_kappa(labels) == 1.0


def test_kappa_is_none_when_every_vote_shared_one_label():
    """Total agreement on a single label is not perfect agreement, it is no evidence.

    Both raters always answering "A" makes expected agreement 1.0, so kappa's
    denominator vanishes. Returning None says "undefined" rather than silently
    reporting a perfect score from a degenerate sample -- which is exactly the state
    the calibration set starts in.
    """
    assert stats.cohens_kappa([("A", "A")] * 4) is None


def test_kappa_is_negative_for_systematic_disagreement():
    labels = [("A", "B")] * 5 + [("B", "A")] * 5
    kappa = stats.cohens_kappa(labels)
    assert kappa is not None and kappa < 0


def test_kappa_on_empty_input_is_none():
    assert stats.cohens_kappa([]) is None
