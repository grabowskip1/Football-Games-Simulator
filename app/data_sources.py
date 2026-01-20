from dataclasses import dataclass
import os
import requests
import pandas as pd
import numpy as np

from app.config import SAVE_DIR, FD_BASE, FD_API_TOKEN, FD_SEASON, today_stamp
from app.leagues import LEAGUES
from app.utils import norm, league_baselines

@dataclass
class LeagueState:
    key: str
    label: str
    df: pd.DataFrame
    teams: list
    HAS_SHOTS: bool
    HAS_CORNERS: bool
    LEAG_AVG_H: float
    LEAG_AVG_A: float
    HFA: float
    standings_df: pd.DataFrame
    rank_lookup: dict
    name_map: dict
    aliases: dict
    table_n: int
    overall: dict
    elo: dict

def _csv_path_for(key: str) -> str:
    cfg = LEAGUES[key]
    return os.path.join(SAVE_DIR, f'{today_stamp()}{cfg['csv_suffix']}.csv')



def _cleanup_old_csv_files(csv_suffix: str, keep_path: str) -> None:
    keep_abs = os.path.abspath(keep_path)
    for filename in os.listdir(SAVE_DIR):
        if not filename.endswith(f'{csv_suffix}.csv'):
            continue
        full_path = os.path.join(SAVE_DIR, filename)
        if os.path.abspath(full_path) == keep_abs:
            continue
        # best-effort cleanup
        try:
            os.remove(full_path)
        except OSError:
            pass


def _ensure_csv(key: str) -> str:
    cfg = LEAGUES[key]
    p = _csv_path_for(key)
    if not os.path.exists(p):
        r = requests.get(cfg['csv_url'], timeout=25)
        if r.status_code != 200:
            raise RuntimeError('CSV download failed')
        with open(p, 'wb') as f:
            f.write(r.content)
        _cleanup_old_csv_files(csv_suffix=cfg['csv_suffix'], keep_path=p)
    return p

def fetch_standings_fd(comp_code: str) -> pd.DataFrame | None:
    if not FD_API_TOKEN:
        return None
    url = f'{FD_BASE}/competitions/{comp_code}/standings'
    headers = {'X-Auth-Token': FD_API_TOKEN}
    params = {'season': FD_SEASON}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
    except Exception:
        return None
    standings = js.get('standings', [])
    total = next((s for s in standings if s.get('type') == 'TOTAL'), None)
    if not total:
        return None
    rows = []
    for e in total.get('table', []) or []:
        pos = e.get('position')
        tn = (e.get('team') or {}).get('name') or (e.get('team') or {}).get('shortName')
        if pos is None or not tn:
            continue
        rows.append({'rank': int(pos), 'team': str(tn)})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('rank').reset_index(drop=True)

def build_elo(d: pd.DataFrame, teams: list, K=20.0, HFA_ELO=60.0) -> dict:
    elo = {t: 1500.0 for t in teams}
    d_iter = d
    if 'Date' in d.columns:
        try:
            dd = d.copy()
            dd['Date'] = pd.to_datetime(dd['Date'], errors='coerce', dayfirst=True)
            d_iter = dd.sort_values('Date')
        except Exception:
            pass
    for _, r in d_iter.iterrows():
        if pd.isna(r['FTHG']) or pd.isna(r['FTAG']):
            continue
        h, a = r['HomeTeam'], r['AwayTeam']
        eh = 1.0 / (1.0 + 10 ** (-(((elo[h] + HFA_ELO) - elo[a]) / 400.0)))
        res = 1.0 if r['FTHG'] > r['FTAG'] else 0.0 if r['FTHG'] < r['FTAG'] else 0.5
        gd = abs(float(r['FTHG']) - float(r['FTAG']))
        mult = 1.0 + np.log1p(gd) * 0.5
        delta = K * mult * (res - eh)
        elo[h] += delta
        elo[a] -= delta
    return elo

def load_league(key: str) -> LeagueState:
    cfg = LEAGUES[key]
    csv_path = _ensure_csv(key)
    df = pd.read_csv(csv_path)

    needed = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'Missing column: {c}')

    teams = sorted(pd.unique(df[['HomeTeam','AwayTeam']].values.ravel('K')))
    HAS_SHOTS = all(c in df.columns for c in ['HS','AS'])
    HAS_CORNERS = all(c in df.columns for c in ['HC','AC'])
    LEAG_AVG_H, LEAG_AVG_A, HFA = league_baselines(df)

    standings_df = fetch_standings_fd(cfg['fd_comp'])
    rank_lookup, table_n = {}, 20
    if standings_df is not None and not standings_df.empty:
        table_n = max(int(standings_df['rank'].max()), 20)
        for _, r_ in standings_df.iterrows():
            rank_lookup[norm(r_['team'])] = int(r_['rank'])

    aliases = cfg['aliases'].copy()
    name_map = {}
    if standings_df is not None and not standings_df.empty:
        api_names = {norm(n): n for n in standings_df['team']}
        for t in teams:
            keyn = norm(t)
            target = aliases.get(keyn)
            if target and norm(target) in api_names:
                name_map[t] = api_names[norm(target)]
            elif keyn in api_names:
                name_map[t] = api_names[keyn]
            else:
                name_map[t] = t
    else:
        for t in teams:
            name_map[t] = t

    overall = {}
    for t in teams:
        gh = df.loc[df['HomeTeam'] == t, 'FTHG'].clip(0, 6)
        ga = df.loc[df['AwayTeam'] == t, 'FTAG'].clip(0, 6)
        ch = df.loc[df['HomeTeam'] == t, 'FTAG'].clip(0, 6)
        ca = df.loc[df['AwayTeam'] == t, 'FTHG'].clip(0, 6)
        scored_avg = pd.concat([gh, ga]).mean() if not pd.concat([gh, ga]).empty else (LEAG_AVG_H + LEAG_AVG_A) / 2
        conceded_avg = pd.concat([ch, ca]).mean() if not pd.concat([ch, ca]).empty else (LEAG_AVG_H + LEAG_AVG_A) / 2
        overall[t] = {'scored_avg': float(max(scored_avg, 0.1)),
                      'conceded_avg': float(max(conceded_avg, 0.1))}

    elo = build_elo(df, teams)

    return LeagueState(
        key=key, label=cfg['label'], df=df, teams=teams,
        HAS_SHOTS=HAS_SHOTS, HAS_CORNERS=HAS_CORNERS,
        LEAG_AVG_H=LEAG_AVG_H, LEAG_AVG_A=LEAG_AVG_A, HFA=HFA,
        standings_df=standings_df if standings_df is not None else pd.DataFrame(),
        rank_lookup=rank_lookup, name_map=name_map, aliases=aliases,
        table_n=table_n, overall=overall, elo=elo
    )
