import numpy as np
import pandas as pd

from wc26.ratings.elo import EloParams, EloResult, expected_score, margin_multiplier, run_elo


def test_expectancy_known_values():
    # eloratings.net: dr=0 -> 0.5; dr=120 -> ~0.666; dr=800 -> ~0.99
    assert expected_score(1500, 1500, 0) == 0.5
    assert abs(expected_score(1620, 1500, 0) - 0.666) < 0.01
    assert expected_score(2300, 1500, 0) > 0.989
    # home advantage shifts expectancy up
    assert expected_score(1500, 1500, 100) > 0.6


def test_margin_multiplier():
    assert margin_multiplier(0, 0.8) == 1.0
    assert margin_multiplier(1, 0.8) == 1.0
    assert margin_multiplier(-1, 0.8) == 1.0
    assert margin_multiplier(2, 0.8) == 2 ** 0.8
    assert margin_multiplier(5, 0.8) > margin_multiplier(3, 0.8)


def test_zero_sum_and_ordering(toy_matches):
    res = run_elo(toy_matches)
    # Elo is zero-sum: total rating mass conserved
    total = sum(res.ratings.values())
    assert abs(total - 1500.0 * 3) < 1e-6
    # A beat everyone, C lost to everyone
    assert res.ratings["A"] > res.ratings["B"] > res.ratings["C"]


def test_history_is_pre_match(toy_matches):
    res = run_elo(toy_matches)
    h = res.history
    # First appearance of every team must show the initial rating (no leakage)
    assert h.iloc[0]["elo_home"] == 1500.0
    assert h.iloc[0]["elo_away"] == 1500.0
    # exp_home in (0,1) always
    assert ((h["exp_home"] > 0) & (h["exp_home"] < 1)).all()


def test_draw_moves_ratings_toward_expectation():
    m = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-01"), "home": "X", "away": "Y",
          "home_goals": 1, "away_goals": 1, "tournament": "Friendly",
          "neutral": True, "tier": "friendly"}]
    )
    # equal teams drawing on neutral ground -> no change
    res = run_elo(m)
    assert abs(res.ratings["X"] - 1500.0) < 1e-9
    assert abs(res.ratings["Y"] - 1500.0) < 1e-9


def test_k_tier_scaling():
    def one(tier):
        m = pd.DataFrame(
            [{"date": pd.Timestamp("2024-01-01"), "home": "X", "away": "Y",
              "home_goals": 1, "away_goals": 0, "tournament": "t",
              "neutral": True, "tier": tier}]
        )
        return run_elo(m).ratings["X"] - 1500.0

    assert one("wc_finals") > one("qualifier") > one("friendly") > 0


def test_rd_grows_with_inactivity():
    m = pd.DataFrame(
        [
            {"date": pd.Timestamp("2018-01-01"), "home": "X", "away": "Y",
             "home_goals": 1, "away_goals": 0, "tournament": "Friendly",
             "neutral": True, "tier": "friendly"},
            {"date": pd.Timestamp("2018-02-01"), "home": "X", "away": "Y",
             "home_goals": 1, "away_goals": 0, "tournament": "Friendly",
             "neutral": True, "tier": "friendly"},
            {"date": pd.Timestamp("2024-01-01"), "home": "X", "away": "Y",
             "home_goals": 1, "away_goals": 0, "tournament": "Friendly",
             "neutral": True, "tier": "friendly"},
        ]
    )
    res = run_elo(m)
    h = res.history
    # after 6 idle years RD at 3rd match must exceed RD at 2nd match
    assert h.iloc[2]["rd_home"] > h.iloc[1]["rd_home"]


def test_rtm_pulls_back_toward_trailing_mean(toy_matches):
    no_rtm = run_elo(toy_matches, EloParams(rtm_weight=1.0))
    rtm = run_elo(toy_matches, EloParams(rtm_weight=0.7))
    # RTM compresses the spread between best and worst
    spread_no = no_rtm.ratings["A"] - no_rtm.ratings["C"]
    spread_rtm = rtm.ratings["A"] - rtm.ratings["C"]
    assert spread_rtm <= spread_no
