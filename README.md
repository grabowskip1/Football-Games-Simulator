# ⚽ Football Match Simulator (Premier League & La Liga)

A data-driven football match simulator that predicts match outcomes
using statistical modeling and real match data.\
It combines historical team performance, league standings, and an
Elo-based strength system to generate realistic simulations.

------------------------------------------------------------------------

## 🧭 System Overview

Below is the core simulation pipeline --- each stage feeds into the next
mathematically:

    ┌────────────────────┐
    │    Data Sources    │
    │ (football-data APIs│
    │ & CSV historicals) │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │  Preprocessing     │
    │  - Normalize names │
    │  - Build averages  │
    │  - Fetch standings │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │  Team Modeling     │
    │  - Attack/Defense  │
    │    EMA ratios      │
    │  - Rank weighting  │
    │  - Elo integration │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │  Expected Goals    │
    │  - λ_h, λ_a calc   │
    │  - Poisson λ mix   │
    │  - Kappa coupling  │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │  Simulation Engine │
    │  - 6000 iterations │
    │  - Bivariate model │
    │  - Noise & scaling │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │   Result Metrics   │
    │   - Score avg/round│
    │   - Possession %   │
    │   - Win/Draw/Away  │
    └────────────────────┘

------------------------------------------------------------------------

## 🧠 Mathematical Summary

  ------------------------------------------------------------------------
  Step            Model                 Description
  --------------- --------------------- ----------------------------------
  **Form          Exponential Moving    Recent matches weigh more than old
  Strength**      Average (EMA)         ones

  **Expected      Poisson parameter     Derived from attack/defense
  Goals (λ)**                           strength + rank + Elo

  **Correlation   Shared Poisson        Models scoring dependency between
  (κ)**           component             teams

  **Outcome       Monte Carlo           6000 draws of (X,Y) from Bivariate
  Sampling**      simulation            Poisson

  **Elo           Logistic expectation  Adjusts λ by long-term strength
  Weighting**                           difference

  **Possession    Normalized            Estimates ball control share
  Model**         shot/corner ratio     
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧮 Example Prediction Flow

1.  Fetch live standings (e.g. Arsenal #1, Wolves #20)\
2.  Load last 10 matches per team\
3.  Compute attack/defense EMAs\
4. Derive expected goals
The expected goals for the home team are computed as:


![formula](https://latex.codecogs.com/svg.image?\dpi{150}\color{White}\lambda_h=\mu_h\times\frac{A_{home}}{D_{away}}\times{B_{rank}}\times{M_{elo}})


6.  Simulate 6000 correlated results\
7.  Aggregate into averages and probabilities

------------------------------------------------------------------------

## 🧩 Features

-   Supports **Premier League (ENG)** and **La Liga (SPA)**\
-   Automatically downloads fresh match data (CSV format)\
-   Integrates **live standings** from *football-data.org*\
-   Calculates team strength using attack, defense, possession & Elo
    rating\
-   Uses **Bivariate Poisson distribution** for goal correlation\
-   Provides a **CustomTkinter GUI** for easy simulation & visualization

------------------------------------------------------------------------

## 🧰 Requirements

Before running the project, install the required dependencies:

``` bash
pip install requests pandas numpy customtkinter
```

Add your **API TOKEN** in the file config.py at:
``` python
FD_API_TOKEN = os.getenv("FD_API_TOKEN", "YOUR_API")
```

------------------------------------------------------------------------

## 🚀 Usage

Run the app with:

``` bash
python main.py
```

Then choose a league (`ENG` or `SPA`), select teams, and click
**Simulate**.\
The results will show predicted scores, win probabilities, and
possession.

------------------------------------------------------------------------

## 🧮 Example Output

    Premier League (ENG)
    Arsenal vs Wolves

    Score (raw): 2.35 - 0.82
    Score (rounded): 2 - 1
    Possession: 61.4% - 38.6%
    P(Home): 68.2%   P(Draw): 18.3%   P(Away): 13.5%
    Table: Arsenal FC #1 | Wolverhampton Wanderers FC #20 (season 2025)

------------------------------------------------------------------------

## ⚽ Example Match Comparison

| Date       | Home Team       | Away Team     | Predicted Score | Predicted Possession | Actual Score | Actual Possession |
|-------------|------------------|----------------|------------------|----------------------|----------------|--------------------|
| 25.02.2025  | Brighton         | Bournemouth    | 2-1              | 54.2 % – 45.8 %     | 2-1            | 44 % – 56 %        |
| 25.02.2025  | Crystal Palace   | Aston Villa    | 1-1              | 44.7 % – 55.3 %     | 4-1            | 36 % – 64 %        |
| 25.02.2025  | Wolves           | Fulham         | 1-2              | 47.9 % – 52.1 %     | 1-2            | 60 % – 40 %        |
| 25.02.2025  | Chelsea          | Southampton    | 3-0              | 58.8 % – 41.2 %     | 4-0            | 60 % – 40 %        |


------------------------------------------------------------------------
