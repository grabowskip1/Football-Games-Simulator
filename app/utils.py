import numpy as np
import pandas as pd

def norm(x: str) -> str:
    return str(x).strip().lower().replace('.', '').replace('-', ' ').replace('&', 'and')

def ema(values, alpha=0.6, default=np.nan):
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return default
    return s.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def league_baselines(d: pd.DataFrame):
    # clip outliers for stability
    lh = float(max(d['FTHG'].clip(0, 6).mean(), 0.1)) if len(d) else 1.4
    la = float(max(d['FTAG'].clip(0, 6).mean(), 0.1)) if len(d) else 1.1
    hfa = (lh / max(la, 1e-6)) ** 0.25
    hfa = float(np.clip(hfa, 0.9, 1.2))
    return lh, la, hfa
