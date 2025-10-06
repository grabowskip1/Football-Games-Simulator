import numpy as np
import pandas as pd
from .utils import ema, norm

# rank-based weights
def team_rank_percentile(team_name: str, state) -> float:
    if not state.rank_lookup:
        return 0.5
    resolved = state.name_map.get(team_name, team_name)
    pos = state.rank_lookup.get(norm(resolved))
    if not pos:
        return 0.5
    denom = max(state.table_n - 1, 1)
    return float(1.0 - (pos - 1) / denom)

def opp_quality_weight(opp_name: str, state) -> float:
    p = team_rank_percentile(opp_name, state)
    return float(0.6 + 0.8 * p)

# possession proxy
def possession_share_row(row, state):
    if state.HAS_SHOTS or state.HAS_CORNERS:
        hs = float(row['HS']) if state.HAS_SHOTS and pd.notna(row['HS']) else 0.0
        a_s = float(row['AS']) if state.HAS_SHOTS and pd.notna(row['AS']) else 0.0
        hc = float(row['HC']) if state.HAS_CORNERS and pd.notna(row['HC']) else 0.0
        ac = float(row['AC']) if state.HAS_CORNERS and pd.notna(row['AC']) else 0.0
        h = hs + 0.8 * hc
        a = a_s + 0.8 * ac
        tot = h + a
        if tot > 0:
            ph = h / tot
            ph = float(np.clip(ph, 0.35, 0.65))
            return ph, 1.0 - ph
    return 0.5, 0.5

# Elo helpers
def elo_multiplier(home_team, away_team, state, scale=800.0):
    eh = state.elo.get(home_team, 1500.0)
    ea = state.elo.get(away_team, 1500.0)
    diff = eh - ea
    m = float(np.exp(diff / scale))
    return float(np.clip(m, 0.6, 1.8))

# team strength
def team_strength(team_name, state, is_home=True, last_matches=10):
    dframe = state.df
    if is_home:
        m = dframe[dframe['HomeTeam'] == team_name].tail(last_matches).copy()
        gf_col, ga_col, opp_col = 'FTHG', 'FTAG', 'AwayTeam'
        h_adv = state.HFA
    else:
        m = dframe[dframe['AwayTeam'] == team_name].tail(last_matches).copy()
        gf_col, ga_col, opp_col = 'FTAG', 'FTHG', 'HomeTeam'
        h_adv = 1.0 / state.HFA

    if m.empty:
        return {'att_rating': 1.0, 'def_rating': 1.0,
                'pos_prior': 0.5,
                'lambda_base': (state.LEAG_AVG_H if is_home else state.LEAG_AVG_A) * h_adv,
                'table_pct': team_rank_percentile(team_name, state)}

    att_ratios, def_ratios, pos_samples, wts = [], [], [], []
    for _, row in m.iterrows():
        opp = row[opp_col]
        opp_def = state.overall.get(opp, {}).get('conceded_avg', (state.LEAG_AVG_H + state.LEAG_AVG_A) / 2)
        opp_att = state.overall.get(opp, {}).get('scored_avg', (state.LEAG_AVG_H + state.LEAG_AVG_A) / 2)
        gf = float(row[gf_col]) if pd.notna(row[gf_col]) else np.nan
        ga = float(row[ga_col]) if pd.notna(row[ga_col]) else np.nan
        w = opp_quality_weight(opp, state) if state.rank_lookup else 1.0
        if pd.notna(gf) and opp_def > 0:
            att_ratios.append(np.log1p(gf) / np.log1p(opp_def))
            wts.append(w)
        if pd.notna(ga) and opp_att > 0:
            def_ratios.append(np.log1p(ga) / np.log1p(opp_att))
        h_share, a_share = possession_share_row(row, state)
        pos_samples.append(h_share if is_home else a_share)

    att_rating = ema(pd.Series(att_ratios) * pd.Series(wts[:len(att_ratios)]) if (att_ratios and wts) else [], alpha=0.55, default=1.0)
    def_rating_raw = ema(def_ratios, alpha=0.55, default=1.0)
    if np.isnan(att_rating): att_rating = 1.0
    if np.isnan(def_rating_raw): def_rating_raw = 1.0
    att_rating = float(np.clip(att_rating, 0.6, 1.6))
    def_rating = float(np.clip(def_rating_raw, 0.6, 1.6))
    pos_prior = float(np.clip(ema(pos_samples, alpha=0.6, default=0.5), 0.35, 0.65))

    base_mu = (state.LEAG_AVG_H if is_home else state.LEAG_AVG_A) * h_adv
    table_pct = team_rank_percentile(team_name, state)

    return {'att_rating': att_rating, 'def_rating': def_rating,
            'pos_prior': pos_prior, 'lambda_base': float(max(base_mu, 0.05)),
            'table_pct': table_pct}

# xG and pace coupling
def expected_goals(home_team, away_team, state):
    hs = team_strength(home_team, state, is_home=True)
    as_ = team_strength(away_team, state, is_home=False)
    lam_h = hs['lambda_base'] * hs['att_rating'] / max(as_['def_rating'], 1e-3)
    lam_a = as_['lambda_base'] * as_['att_rating'] / max(hs['def_rating'], 1e-3)

    if state.rank_lookup:
        boost_h = 0.7 + 0.6 * (hs['table_pct'] ** 1.5)
        boost_a = 0.7 + 0.6 * (as_['table_pct'] ** 1.5)
        lam_h *= boost_h / max(boost_a, 1e-6)
        lam_a *= boost_a / max(boost_h, 1e-6)

    em = elo_multiplier(home_team, away_team, state, scale=800.0)
    lam_h *= em
    lam_a /= em

    lam_h = float(np.clip(lam_h, 0.15, 3.8))
    lam_a = float(np.clip(lam_a, 0.15, 3.8))

    ph, pa = float(hs['pos_prior']), float(as_['pos_prior'])
    s = ph + pa
    if s > 0:
        ph = ph / s
        pa = 1.0 - ph
    else:
        ph, pa = 0.5, 0.5

    pace = 1.0
    if state.HAS_SHOTS or state.HAS_CORNERS:
        ls = (state.df.get('HS', pd.Series(0)).fillna(0) + state.df.get('AS', pd.Series(0)).fillna(0) +
              0.8 * (state.df.get('HC', pd.Series(0)).fillna(0) + state.df.get('AC', pd.Series(0)).fillna(0)))
        m = ls.replace(0, np.nan).mean()
        pace = float(np.clip((ls.mean() / max(m, 1e-6)), 0.8, 1.2))
    kappa = float(np.clip(0.18 * min(lam_h, lam_a) * pace, 0.0, min(lam_h, lam_a) * 0.49))

    return lam_h, lam_a, ph, pa, kappa

# bivariate Poisson
_rng = np.random.default_rng(42)
def _sample_bivar_poisson(l1, l2, k):
    l1p = max(l1 - k, 1e-8)
    l2p = max(l2 - k, 1e-8)
    x = _rng.poisson(l1p)
    y = _rng.poisson(l2p)
    z = _rng.poisson(k)
    return x + z, y + z

def simulate_match(home_team, away_team, state, n=6000):
    lam_h, lam_a, ph, pa, kappa = expected_goals(home_team, away_team, state)
    hg = np.empty(n, dtype=int); ag = np.empty(n, dtype=int)
    for i in range(n):
        h, a = _sample_bivar_poisson(lam_h, lam_a, kappa)
        hg[i], ag[i] = h, a

    p_noise_h = np.clip(ph + _rng.normal(0, 0.03, size=n), 0.01, 0.99)
    p_noise_a = np.clip(pa + _rng.normal(0, 0.03, size=n), 0.01, 0.99)
    s = p_noise_h + p_noise_a
    p_h = p_noise_h / s
    p_a = 1.0 - p_h

    low_mask = (hg <= 1) & (ag <= 1)
    if low_mask.any():
        boost = 0.07
        draw_idx = np.where((hg == ag) & low_mask)[0]
        flip_idx = np.where((hg != ag) & low_mask)[0]
        k_adj = int(min(len(flip_idx), np.floor(len(draw_idx) * boost)))
        if k_adj > 0:
            take = _rng.choice(flip_idx, size=k_adj, replace=False)
            for j in take:
                if hg[j] > ag[j]:
                    ag[j] = hg[j]
                else:
                    hg[j] = ag[j]

    return {
        'home_goals_avg': float(np.mean(hg)),
        'away_goals_avg': float(np.mean(ag)),
        'home_goals_round': int(np.round(np.mean(hg))),
        'away_goals_round': int(np.round(np.mean(ag))),
        'home_pos_pct': float(np.mean(p_h) * 100.0),
        'away_pos_pct': float(np.mean(p_a) * 100.0),
        'win_p_home': float(np.mean(hg > ag)),
        'win_p_away': float(np.mean(ag > hg)),
        'draw_p': float(np.mean(hg == ag))
    }
