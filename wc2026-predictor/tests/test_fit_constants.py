import pandas as pd

from synthetic import make_matches
from wc26.ratings.elo import EloParams
from wc26.ratings.fit import fit_elo_params


def test_fit_elo_params_smoke(tmp_path):
    matches, _ = make_matches(n=1500, seed=9)
    out = tmp_path / "elo_params.json"
    fitted, ll = fit_elo_params(
        matches, eval_start=matches["date"].iloc[900], out_path=out, maxfev=8
    )
    # objective is a real log-loss (uniform = ln 3 ~ 1.0986)
    assert 0 < ll < 1.2
    # constraints respected
    assert 0.0 <= fitted.home_advantage <= 300.0
    assert 0.0 <= fitted.margin_exponent <= 2.0
    assert 0.5 <= fitted.rtm_weight <= 1.0
    assert out.exists()


def test_fitted_no_worse_than_defaults():
    matches, _ = make_matches(n=1500, seed=9)
    from wc26.ratings.fit import _objective

    eval_start = matches["date"].iloc[900]
    base_ll = _objective(matches, EloParams(), eval_start)
    fitted, ll = fit_elo_params(matches, eval_start=eval_start, maxfev=30)
    assert ll <= base_ll + 1e-6
