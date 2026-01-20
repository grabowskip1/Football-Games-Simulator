# ⚽ Football Match Simulator (Premier League & La Liga)

A data-driven football match simulator written in Python.
The simulator predicts football match outcomes using statistical modeling
based on historical match data, league standings, and team strength indicators.
Match results are generated using an Expected Goals (xG) framework combined
with Poisson-based score simulation, producing realistic scorelines,
probabilities, and possession estimates.

------------------------------------------------------------------------

## 🧭 System Overview

Below is the actual simulation pipeline used in the project. Each stage feeds
into the next and is directly reflected in the codebase:
```
┌────────────────────┐
│    Data Sources    │
│ (football-data API │
│  & cached CSVs)    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Preprocessing     │
│ - Normalize teams  │
│ - Load standings   │
│ - Match filtering  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Team Modeling     │
│ - Attack strength  │
│ - Defense strength │
│ - EMA-based form   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Expected Goals    │
│ - λ_home, λ_away   │
│ - Strength ratios  │
│ - Home advantage   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Simulation Engine │
│ - Poisson draws    │
│ - Monte Carlo avg  │
│ - Outcome probs    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Result Metrics   │
│ - Avg score        │
│ - Win probabilities│
│ - Possession       │
└────────────────────┘
```

------------------------------------------------------------------------

## 🧠 Mathematical Summary

    ---------------------------------------------------------------------------
    Component          Model                     Description
    ------------------ ------------------------- --------------------------------
    Recent Form        Exponential Moving Avg    Recent matches influence team
                                                strength more than older ones

    Attack / Defense   Ratio-based strength      Goals scored vs conceded,
    Strength                                     smoothed with EMA

    Expected Goals     Poisson mean (λ)          Derived from attack/defense
    (xG)                                         ratios and home advantage

    Score Sampling     Poisson distribution      Goals sampled as rare events
                                                using λ_home and λ_away

    Outcome Metrics    Monte Carlo aggregation   Averages and probabilities over
                                                many simulated matches

    Possession Model   Normalized shot ratios    Estimates possession share based
                                                on offensive activity
    ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧮 Example Prediction Flow

1.  Download or load cached historical match data (CSV)\
2.  Load live league standings\
3.  Compute attack/defense EMA\
4. Derive expected goals
The expected goals for the home team are computed as:


![formula](https://latex.codecogs.com/svg.image?\dpi{150}\color{White}\lambda_h=\mu_h\times\frac{A_{home}}{D_{away}}\times{B_{rank}}\times{M_{elo}})



5.  Simulate match scores using Poisson sampling\
6.  Aggregate results into averages and probabilities

------------------------------------------------------------------------

## 🧩 Features

-   Supports **Premier League (ENG)** and **La Liga (SPA)**\
-   Automatically downloads fresh match data (CSV format)\
-   Integrates **live standings** from *football-data.org*\
-   Calculates team strength using attack, defense, possession & Elo
    rating\
-   Uses **EMA-based attack and defense strength modeling**\
-   **Expected Goals (xG)** computed as Poisson parameters\
-   Monte Carlo simulation for score and probability estimation\
-   Predicts: **scores, win/draw/lose probabilities, ball possesion**\
-   Provides a **CustomTkinter GUI** for easy simulation & visualization

------------------------------------------------------------------------

## 🗂️ Project Structure

engine/
  match_engine.py     # core match simulation logic (xG + Poisson)

app/
  data_sources.py     # data fetching and CSV caching
  model.py            # wrapper around the match engine
  ui.py               # CLI output formatting

tests/
  test_match_engine.py

------------------------------------------------------------------------

## 🧰 Requirements

Before running the project, install the required dependencies:

``` bash
pip install requests pandas numpy customtkinter pytest
```

Add your **API TOKEN** in the file app/config.py at:
``` python
FD_API_TOKEN = os.getenv("FD_API_TOKEN", "YOUR_API_KEY")
```

------------------------------------------------------------------------

## 🧪 Testing
The match simulation engine is covered by unit tests using pytest.
Tests include:
validation of expected goal parameters
statistical sanity checks (stronger teams win more often over many simulations)

Run tests with:
```bash
pytest
```

------------------------------------------------------------------------

## 🚀 Usage

Run the app with:

```bash
python main.py
```
Then choose a league (`ENG` or `SPA`), select teams, and click
**Simulate**.

------------------------------------------------------------------------

## 🧮 Example Output

    Premier League (ENG)
    Man United vs Brighton

    Score (raw): 2.51 - 1.33
    Score (rounded): 3 - 1
    xG: 2.51 - 1.20 (Poisson)
    Possession: 55.1% - 44.9%
    P(Home): 61.4% | P(Draw): 25.4% | P(Away): 13.2%
    Table: Manchester United FC #7 | Brighton & Hove Albion FC #11

------------------------------------------------------------------------

## ⚽ Real Match Comparison Simulations

| Date       | Home Team       | Away Team     | Predicted Score | Predicted Possession | Actual Score | Actual Possession |
|-------------|------------------|----------------|------------------|----------------------|----------------|--------------------|
| 25.02.2025  | Brighton         | Bournemouth    | 2-1              | 54.2 % – 45.8 %     | 2-1            | 44 % – 56 %        |
| 25.02.2025  | Crystal Palace   | Aston Villa    | 1-1              | 44.7 % – 55.3 %     | 4-1            | 36 % – 64 %        |
| 25.02.2025  | Wolves           | Fulham         | 1-2              | 47.9 % – 52.1 %     | 1-2            | 60 % – 40 %        |
| 25.02.2025  | Chelsea          | Southampton    | 3-0              | 58.8 % – 41.2 %     | 4-0            | 60 % – 40 %        |

*More results will be added after further testing.*
------------------------------------------------------------------------
