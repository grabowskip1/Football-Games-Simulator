'''Backward-compatible facade for the match engine.
Historically the project exposed `simulate_match()` from `app.model`.
After the refactor, the implementation lives in `engine.match_engine`.
This module keeps the public API stable for the UI.
'''
from __future__ import annotations
from typing import Dict, Any
from engine.match_engine import MatchEngine

# single engine instance (deterministic by default)
_ENGINE = MatchEngine(rng_seed=42)

def simulate_match(home_team: str, away_team: str, state, n: int = 6000) -> Dict[str, Any]:
    '''Simulate a match and return the legacy dict, with extra explainability fields.'''
    res = _ENGINE.simulate_match(home_team, away_team, state, n=n)
    expl = res.explanation

    out: Dict[str, Any] = {
        # legacy
        'home_goals_avg': res.home_goals_avg,
        'away_goals_avg': res.away_goals_avg,
        'home_goals_round': res.home_goals_round,
        'away_goals_round': res.away_goals_round,
        'home_pos_pct': res.home_pos_pct,
        'away_pos_pct': res.away_pos_pct,
        'win_p_home': res.win_p_home,
        'win_p_away': res.win_p_away,
        'draw_p': res.draw_p,

        'xg_home': expl.lambda_home,
        'xg_away': expl.lambda_away,
        'kappa': expl.kappa,
        'explanation': {
            'base_home': expl.base_home,
            'base_away': expl.base_away,
            'att_home': expl.att_home,
            'def_home': expl.def_home,
            'att_away': expl.att_away,
            'def_away': expl.def_away,
            'rank_boost_home': expl.rank_boost_home,
            'rank_boost_away': expl.rank_boost_away,
            'elo_multiplier': expl.elo_multiplier,
            'pace': expl.pace,
            'pos_prior_home': expl.pos_prior_home,
            'pos_prior_away': expl.pos_prior_away,
        }
    }
    return out
