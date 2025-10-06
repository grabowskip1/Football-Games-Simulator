import tkinter as tk
import customtkinter as ctk

from app.leagues import LEAGUES
from app.data_sources import load_league, LeagueState
from app.model import simulate_match
from app.utils import norm

# --- App-level state (selected league) ---
_state: LeagueState | None = None


# --- Textbox helpers ---
def _write_result(widget: ctk.CTkTextbox, txt: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", txt)
    widget.configure(state="disabled")


# --- Rank lookup with aliases + fuzzy fallback ---
def _get_rank(team_name: str, state: LeagueState) -> int | None:
    r = state.rank_lookup.get(norm(team_name))
    if r is not None:
        return r
    alias = state.aliases.get(norm(team_name))
    if alias:
        r = state.rank_lookup.get(norm(alias))
        if r is not None:
            return r
    if state.standings_df is not None and not state.standings_df.empty:
        tnorm = norm(team_name)
        col = state.standings_df["team"].astype(str)
        mask = col.apply(lambda x: norm(x) == tnorm)
        if mask.any():
            return int(state.standings_df.loc[mask, "rank"].iloc[0])
        toks = [tok for tok in tnorm.split() if tok]
        if toks:
            mask = col.apply(lambda x: all(tok in norm(x) for tok in toks))
            if mask.any():
                return int(state.standings_df.loc[mask, "rank"].iloc[0])
    return None


# --- Main window ---
def run_app() -> None:
    global _state

    # theme + DPI scaling
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    ctk.set_widget_scaling(1.25)
    ctk.set_window_scaling(1.15)

    # root
    root = ctk.CTk()
    root.title("Match Simulator")
    root.geometry("900x720")
    root.minsize(860, 680)

    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # --- Header layout ---
    header = ctk.CTkFrame(root, corner_radius=14)
    header.grid(row=0, column=0, sticky="ew", padx=18, pady=14)
    header.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

    title = ctk.CTkLabel(
        header,
        text="Football Match Simulator",
        font=ctk.CTkFont(size=28, weight="bold"),
    )
    title.grid(row=0, column=0, columnspan=6, sticky="w", padx=14, pady=(8, 6))

    # league selector
    league_var = tk.StringVar(value="ENG")
    league_label = ctk.CTkLabel(header, text="League")
    league_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))

    league_cb = ctk.CTkComboBox(
        header,
        values=list(LEAGUES.keys()),
        variable=league_var,
        width=140,
    )
    league_cb.grid(row=1, column=1, sticky="w", padx=8, pady=(0, 8))

    run_btn = ctk.CTkButton(header, text="Simulate", width=170, height=36)
    run_btn.grid(row=1, column=5, sticky="e", padx=14, pady=(0, 8))

    # teams row
    home_var = tk.StringVar()
    away_var = tk.StringVar()

    home_label = ctk.CTkLabel(header, text="Home")
    home_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))
    home_cb = ctk.CTkComboBox(header, variable=home_var, values=[], width=280)
    home_cb.grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=(0, 10))

    away_label = ctk.CTkLabel(header, text="Away")
    away_label.grid(row=2, column=3, sticky="w", padx=14, pady=(0, 10))
    away_cb = ctk.CTkComboBox(header, variable=away_var, values=[], width=280)
    away_cb.grid(row=2, column=4, columnspan=2, sticky="w", padx=8, pady=(0, 10))

    meta = ctk.CTkLabel(
        header,
        text="Source: football-data.org, football-data.co.uk",
        text_color=("gray70"),
    )
    meta.grid(row=3, column=0, columnspan=6, sticky="w", padx=14, pady=(0, 4))

    # --- Results panel ---
    content = ctk.CTkFrame(root, corner_radius=14)
    content.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)

    result_box = ctk.CTkTextbox(
        content, wrap="word", activate_scrollbars=True, font=ctk.CTkFont(size=16)
    )
    result_box.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
    result_box.configure(state="disabled")

    # --- Handlers ---
    def on_league_change() -> None:
        global _state
        key = league_var.get()
        _state = load_league(key)
        home_cb.configure(values=_state.teams)
        away_cb.configure(values=_state.teams)
        if _state.teams:
            home_var.set(_state.teams[0])
            away_var.set(_state.teams[-1])
        _write_result(
            result_box,
            f"Loaded {_state.label}. Teams: {len(_state.teams)}. "
            f"Standings rows: {0 if _state.standings_df is None else len(_state.standings_df)}.",
        )

    def predict() -> None:
        if _state is None:
            _write_result(result_box, "Select league first.")
            return
        h = home_var.get()
        a = away_var.get()
        if not h or not a:
            _write_result(result_box, "Select both teams.")
            return
        out = simulate_match(h, a, _state, n=6000)
        text = (
            f"{_state.label}\n"
            f"{h} vs {a}\n"
            f"Score (raw): {out['home_goals_avg']:.2f}-{out['away_goals_avg']:.2f}\n"
            f"Score (rounded): {out['home_goals_round']}-{out['away_goals_round']}\n"
            f"Possession: {out['home_pos_pct']:.1f}% - {out['away_pos_pct']:.1f}%\n"
            f"P(Home): {out['win_p_home']*100:.1f}%   "
            f"P(Draw): {out['draw_p']*100:.1f}%   "
            f"P(Away): {out['win_p_away']*100:.1f}%"
        )
        if _state.standings_df is not None and not _state.standings_df.empty:
            hn = _state.name_map.get(h, h)
            an = _state.name_map.get(a, a)
            hr = _get_rank(hn, _state)
            ar = _get_rank(an, _state)
            if hr is not None and ar is not None:
                text += f"\nTable: {hn} #{hr}  |  {an} #{ar}"
            else:
                text += "\n(Table data missing for selected team.)"
        _write_result(result_box, text)

    # bind actions
    league_cb.configure(command=lambda _: on_league_change())
    run_btn.configure(command=predict)

    # initial load
    on_league_change()
    root.mainloop()
