import pandas as pd

from app.data_sources import LeagueState
from engine.match_engine import MatchEngine


def _dummy_state() -> LeagueState:
    # minimal historical matches for two teams
    df = pd.DataFrame([
        {'HomeTeam':'A','AwayTeam':'B','FTHG':2,'FTAG':0,'HS':10,'AS':6,'HC':5,'AC':3},
        {'HomeTeam':'B','AwayTeam':'A','FTHG':1,'FTAG':1,'HS':8,'AS':9,'HC':4,'AC':4},
        {'HomeTeam':'A','AwayTeam':'B','FTHG':3,'FTAG':1,'HS':12,'AS':5,'HC':6,'AC':2},
        {'HomeTeam':'B','AwayTeam':'A','FTHG':0,'FTAG':2,'HS':7,'AS':11,'HC':3,'AC':6},
    ])

    overall = {
        'A': {'scored_avg': 2.0, 'conceded_avg': 0.75},
        'B': {'scored_avg': 0.75, 'conceded_avg': 2.0},
    }

    # stronger elo for A
    elo = {'A': 1650.0, 'B': 1350.0}

    return LeagueState(
        key='TEST',
        label='Test League',
        df=df,
        teams=['A','B'],
        HAS_SHOTS=True,
        HAS_CORNERS=True,
        LEAG_AVG_H=1.45,
        LEAG_AVG_A=1.15,
        HFA=1.08,
        standings_df=pd.DataFrame(),
        rank_lookup={},   # keep empty to avoid needing standings parsing
        name_map={},
        aliases={},
        table_n=2,
        overall=overall,
        elo=elo,
    )


def test_simulate_match_returns_explainability_fields():
    state = _dummy_state()
    eng = MatchEngine(rng_seed=123)
    res = eng.simulate_match('A','B',state,n=2000)
    assert res.explanation.lambda_home > 0
    assert res.explanation.lambda_away > 0
    assert 0 <= res.win_p_home <= 1
    assert 0 <= res.draw_p <= 1
    assert abs((res.win_p_home + res.draw_p + res.win_p_away) - 1.0) < 0.05


def test_stronger_team_wins_more_often():
    state = _dummy_state()
    eng = MatchEngine(rng_seed=7)
    res = eng.simulate_match('A','B',state,n=6000)
    assert res.win_p_home > 0.55
