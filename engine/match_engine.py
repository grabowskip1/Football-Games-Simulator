from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.utils import ema, norm
from app.data_sources import LeagueState


@dataclass(frozen=True)
class MatchExplanation:
    # 'xG' intensities before sampling
    lambda_home: float
    lambda_away: float

    # possession priors (used as a soft modifier in the existing model)
    pos_prior_home: float
    pos_prior_away: float

    # shared-goal component for bivariate Poisson
    kappa: float

    # breakdown
    base_home: float
    base_away: float
    att_home: float
    def_home: float
    att_away: float
    def_away: float
    rank_boost_home: float
    rank_boost_away: float
    elo_multiplier: float
    pace: float


@dataclass(frozen=True)
class MatchResult:
    home_team: str
    away_team: str

    # Monte Carlo aggregates
    home_goals_avg: float
    away_goals_avg: float
    home_goals_round: int
    away_goals_round: int

    home_pos_pct: float
    away_pos_pct: float

    win_p_home: float
    win_p_away: float
    draw_p: float

    explanation: MatchExplanation


class MatchEngine:
    '''Match engine extracted from app.model with explicit explainability.

    Contract:
      - Pure function over (home_team, away_team, LeagueState)
      - Returns MatchResult with both probabilities and model breakdown.
    '''

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = np.random.default_rng(rng_seed)

    # rank-based weights
    @staticmethod
    def _team_rank_percentile(team_name: str, state: LeagueState) -> float:
        if not state.rank_lookup:
            return 0.5
        resolved = state.name_map.get(team_name, team_name)
        pos = state.rank_lookup.get(norm(resolved))
        if not pos:
            return 0.5
        denom = max(state.table_n - 1, 1)
        return float(1.0 - (pos - 1) / denom)

    @staticmethod
    def _opp_quality_weight(opp_name: str, state: LeagueState) -> float:
        p = MatchEngine._team_rank_percentile(opp_name, state)
        return float(0.6 + 0.8 * p)

    # possession proxy
    @staticmethod
    def _possession_share_row(row: pd.Series, state: LeagueState) -> Tuple[float, float]:
        if state.HAS_SHOTS or state.HAS_CORNERS:
            hs = float(row['HS']) if state.HAS_SHOTS and pd.notna(row.get('HS')) else 0.0
            a_s = float(row['AS']) if state.HAS_SHOTS and pd.notna(row.get('AS')) else 0.0
            hc = float(row['HC']) if state.HAS_CORNERS and pd.notna(row.get('HC')) else 0.0
            ac = float(row['AC']) if state.HAS_CORNERS and pd.notna(row.get('AC')) else 0.0
            h = hs + 0.8 * hc
            a = a_s + 0.8 * ac
            tot = h + a
            if tot > 0:
                ph = h / tot
                ph = float(np.clip(ph, 0.35, 0.65))
                return ph, 1.0 - ph
        return 0.5, 0.5

    # elo helpers
    @staticmethod
    def _elo_multiplier(home_team: str, away_team: str, state: LeagueState, scale: float = 800.0) -> float:
        eh = float(state.elo.get(home_team, 1500.0))
        ea = float(state.elo.get(away_team, 1500.0))
        diff = eh - ea
        m = float(np.exp(diff / scale))
        return float(np.clip(m, 0.6, 1.8))

    # team strength
    def _team_strength(self, team_name: str, state: LeagueState, *, is_home: bool = True, last_matches: int = 10) -> Dict[str, float]:
        dframe = state.df

        if is_home:
            m = dframe[dframe['HomeTeam'] == team_name].tail(last_matches).copy()
            gf_col, ga_col, opp_col = 'FTHG', 'FTAG', 'AwayTeam'
            h_adv = float(state.HFA)
            base = float(state.LEAG_AVG_H) * h_adv
        else:
            m = dframe[dframe['AwayTeam'] == team_name].tail(last_matches).copy()
            gf_col, ga_col, opp_col = 'FTAG', 'FTHG', 'HomeTeam'
            h_adv = float(1.0 / state.HFA)
            base = float(state.LEAG_AVG_A) * h_adv

        table_pct = self._team_rank_percentile(team_name, state)

        if m.empty:
            return {
                'att_rating': 1.0,
                'def_rating': 1.0,
                'pos_prior': 0.5,
                'lambda_base': base,
                'table_pct': table_pct,
                'pace': 1.0,
            }

        att_ratios, def_ratios, pos_samples, pace_samples, wts = [], [], [], [], []
        league_goal_avg = (float(state.LEAG_AVG_H) + float(state.LEAG_AVG_A)) / 2.0

        for _, row in m.iterrows():
            opp = row[opp_col]
            opp_stats = state.overall.get(opp, {})
            opp_def = float(opp_stats.get('conceded_avg', league_goal_avg))
            opp_att = float(opp_stats.get('scored_avg', league_goal_avg))

            gf = float(row[gf_col]) if pd.notna(row[gf_col]) else np.nan
            ga = float(row[ga_col]) if pd.notna(row[ga_col]) else np.nan

            ph, pa = self._possession_share_row(row, state)
            pos = ph if is_home else pa

            # more total shots/corners => more eventful games
            pace = 1.0
            if state.HAS_SHOTS and pd.notna(row.get('HS')) and pd.notna(row.get('AS')):
                pace *= 1.0 + 0.02 * float(row['HS'] + row['AS'])
            if state.HAS_CORNERS and pd.notna(row.get('HC')) and pd.notna(row.get('AC')):
                pace *= 1.0 + 0.01 * float(row['HC'] + row['AC'])
            pace = float(np.clip(pace, 0.75, 1.6))

            w = self._opp_quality_weight(str(opp), state)
            wts.append(w)

            if not np.isnan(gf):
                # scoring vs opponent defense
                att_ratios.append(max(gf, 0.0) / max(opp_def, 0.25))
            if not np.isnan(ga):
                # conceding vs opponent attack
                def_ratios.append(max(ga, 0.0) / max(opp_att, 0.25))

            pos_samples.append(pos)
            pace_samples.append(pace)

        wts_arr = np.asarray(wts, dtype=float) if wts else np.asarray([1.0], dtype=float)

        # smooth ratios and shrink
        att_raw = float(np.average(att_ratios, weights=wts_arr[:len(att_ratios)]) if att_ratios else 1.0)
        def_raw = float(np.average(def_ratios, weights=wts_arr[:len(def_ratios)]) if def_ratios else 1.0)

        # EMA over last games
        att_sm = float(ema(att_ratios, alpha=0.35, default=att_raw))
        def_sm = float(ema(def_ratios, alpha=0.35, default=def_raw))

        # shrink extreme values
        att_rating = float(np.clip(0.55 + 0.45 * att_sm, 0.65, 1.6))
        def_rating = float(np.clip(0.55 + 0.45 * def_sm, 0.65, 1.6))

        pos_prior = float(np.clip(np.average(pos_samples, weights=wts_arr), 0.40, 0.60))
        pace = float(np.clip(np.average(pace_samples, weights=wts_arr), 0.85, 1.35))

        return {
            'att_rating': att_rating,
            'def_rating': def_rating,
            'pos_prior': pos_prior,
            'lambda_base': base,
            'table_pct': table_pct,
            'pace': pace,
        }

    def expected_goals(self, home_team: str, away_team: str, state: LeagueState) -> MatchExplanation:
        hs = self._team_strength(home_team, state, is_home=True)
        as_ = self._team_strength(away_team, state, is_home=False)

        base_h = float(hs['lambda_base'])
        base_a = float(as_['lambda_base'])

        # base xG using attack/defense ratios
        lam_h = base_h * float(hs['att_rating']) / max(float(as_['def_rating']), 1e-3)
        lam_a = base_a * float(as_['att_rating']) / max(float(hs['def_rating']), 1e-3)

        rank_boost_h = 1.0
        rank_boost_a = 1.0
        if state.rank_lookup:
            boost_h = 0.7 + 0.6 * (float(hs['table_pct']) ** 1.5)
            boost_a = 0.7 + 0.6 * (float(as_['table_pct']) ** 1.5)
            # relative boost so both sides are comparable
            rank_boost_h = float(boost_h / max(boost_a, 1e-6))
            rank_boost_a = float(boost_a / max(boost_h, 1e-6))
            lam_h *= rank_boost_h
            lam_a *= rank_boost_a

        elo_m = self._elo_multiplier(home_team, away_team, state, scale=800.0)
        lam_h *= elo_m
        lam_a /= elo_m

        ph = float(hs['pos_prior'])
        pa = float(as_['pos_prior'])

        # pace affects both teams
        pace = float(np.clip(0.5 * (float(hs['pace']) + float(as_['pace'])), 0.85, 1.30))
        lam_h *= pace
        lam_a *= pace

        # avoid absurd scorelines
        lam_h = float(np.clip(lam_h, 0.2, 3.2))
        lam_a = float(np.clip(lam_a, 0.2, 3.2))

        # shared component (low)
        kappa = float(np.clip(0.18 * min(lam_h, lam_a) * pace, 0.0, min(lam_h, lam_a) * 0.49))

        return MatchExplanation(
            lambda_home=lam_h,
            lambda_away=lam_a,
            pos_prior_home=ph,
            pos_prior_away=pa,
            kappa=kappa,
            base_home=base_h,
            base_away=base_a,
            att_home=float(hs['att_rating']),
            def_home=float(hs['def_rating']),
            att_away=float(as_['att_rating']),
            def_away=float(as_['def_rating']),
            rank_boost_home=rank_boost_h,
            rank_boost_away=rank_boost_a,
            elo_multiplier=float(elo_m),
            pace=pace,
        )

    def _sample_bivariate_poisson(self, l1: float, l2: float, k: float) -> Tuple[int, int]:
        l1p = max(l1 - k, 1e-8)
        l2p = max(l2 - k, 1e-8)
        x = int(self._rng.poisson(l1p))
        y = int(self._rng.poisson(l2p))
        z = int(self._rng.poisson(k))
        return x + z, y + z

    def simulate_match(self, home_team: str, away_team: str, state: LeagueState, n: int = 6000) -> MatchResult:
        expl = self.expected_goals(home_team, away_team, state)

        hg = np.empty(n, dtype=int)
        ag = np.empty(n, dtype=int)
        for i in range(n):
            h, a = self._sample_bivariate_poisson(expl.lambda_home, expl.lambda_away, expl.kappa)
            hg[i], ag[i] = h, a

        # possession as soft modifier to outcome
        p_noise_h = np.clip(expl.pos_prior_home + self._rng.normal(0, 0.03, size=n), 0.01, 0.99)
        p_noise_a = np.clip(expl.pos_prior_away + self._rng.normal(0, 0.03, size=n), 0.01, 0.99)
        s = p_noise_h + p_noise_a
        p_h = p_noise_h / s
        p_a = 1.0 - p_h

        # occasional 'dominance flip' to avoid deterministic possession
        flip = self._rng.random(n) < 0.08
        for j in np.where(flip)[0]:
            if hg[j] != ag[j]:
                if hg[j] > ag[j]:
                    ag[j] = hg[j]
                else:
                    hg[j] = ag[j]

        return MatchResult(
            home_team=home_team,
            away_team=away_team,
            home_goals_avg=float(np.mean(hg)),
            away_goals_avg=float(np.mean(ag)),
            home_goals_round=int(np.round(np.mean(hg))),
            away_goals_round=int(np.round(np.mean(ag))),
            home_pos_pct=float(np.mean(p_h) * 100.0),
            away_pos_pct=float(np.mean(p_a) * 100.0),
            win_p_home=float(np.mean(hg > ag)),
            win_p_away=float(np.mean(ag > hg)),
            draw_p=float(np.mean(hg == ag)),
            explanation=expl,
        )
