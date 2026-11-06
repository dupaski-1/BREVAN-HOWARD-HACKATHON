# Multi-asset allocation agent (BREVAN HOWARD HACKATHON EPFL 2026) - 3rd Place 🥉

My agent (team Neyman) for part 2 of the Uncertain World Simulator hackathon, November 2026. It runs a weekly book across 5 asset classes inside a fixed investment mandate, and has to avoid losing that mandate through drawdowns or broken limits.

## The setting

Each week the agent sees lagged returns for 5 asset classes plus a few macro signals (growth, inflation, volatility), and chooses how much of its capital to hold in each one. Whatever it leaves out stays in cash.

| Class | Asset | Reference | Band |
|---|---|---|---|
| A | Developed equity | 35 | 0 to 60 |
| B | Emerging equity | 10 | 0 to 20 |
| C | US 10-year Treasury | 35 | 15 to 55 |
| D | Investment-grade credit | 15 | 0 to 30 |
| E | Oil (WTI futures) | 5 | 0 to 10 |

Rules of the mandate:
* A and B together stay between 20 and 60 units (1 unit is 1 % of capital)
* at most 30 units in cash
* a capped turnover per week
* a 104-week warm-up, after which a drawdown of 20 % loses the mandate

## How the agent decides

Every week runs through the same pipeline in `decide()`:

1. **Risk budget.** Estimates the covariance of the 5 classes from the last year of returns and solves for weights where each class carries a set share of total risk (little for oil, whose futures roll drags on returns). The book is then scaled to a 7 % annual volatility target.
2. **Regime view.** Reads growth and inflation signals and assigns probabilities to 4 macro regimes: goldilocks, inflation, recession and stagflation. Confidence depends on how far the signals sit from the cut-offs.
3. **Learnt returns by regime.** Tracks the average return of each class in each regime the agent called, shrunk toward the overall average until enough weeks have been seen.
4. **Tilt.** Moves the book up to 8 units per class, based on a mix of the learnt returns (per unit of risk), textbook regime priors, and 13-week momentum. The regime probabilities also blend in market-led odds driven by volatility, since markets tend to move before the macro data does.
5. **Keeping the mandate.** Applied last, so no tilt can undo it. As drawdown grows from 5 % toward 13 %, the book rotates into a defensive allocation. When volatility rises above its usual (median) level, every class is shrunk by the same factor.
6. **Trading.** Ignores small changes below a per-class threshold to save turnover, but trades at once when a real risk cut is needed. Every order is projected back inside the bands, the equity limit and the cash limit.

The agent also reports its regime view as weighted scenarios, as the competition requires.

## Team

Team Neyman brought together people from very different backgrounds, which made it a lot of fun:

* Arnaud Du Pasquier ([@dupaski-1](https://github.com/dupaski-1))
* Grégoire Du Pasquier (https://www.linkedin.com/in/gregoire-du-pasquier-68104412b/)
* Tianer Zhou (https://www.linkedin.com/in/tianer-zhou/, https://github.com/ezhoureal)
* Marco D'Addezio (https://www.linkedin.com/in/marco-d-addezio-2bb3049a/)

## Running it

`agent.py` depends on the `uws` package provided by the organisers, which is not included here, so the file does not run on its own.

## Results

**3rd place overall**, across all parts of the competition and the final presentation.

* At the 17:00 leaderboard snapshot: **rank 2 of 15**, +3.97 % a year, 0 of 4 mandates lost
* Final scored agent: +3.53 % a year, 3 of 4 mandates kept, 4 of 5 modules validated (20 of 25 points)
* For comparison, the organisers' reference book lost 3 of 4 mandates with a worst drawdown of 36.8 %
* Turnover of 1.44x and costs of 0.089 % a year, well inside the limits

## What went wrong

The final agent lost the 2018 to 2026 window in October 2022, with a drawdown of 21.4 %. As the drawdown grew, the agent rotated into its defensive book, which held 50 units in US Treasuries. In 2022 inflation drove bonds and equities down together, so the "safe" asset was part of the loss.

The code already had a check for this (`_bonds_move_with_equities`), but `DEFENSIVE_NO_BONDS` ended up identical to `DEFENSIVE` after an earlier credit and oil version lost 2 mandates in testing. The switch therefore had no effect.

The agent also reported a constant confidence every week and never escalated, which scored 0 on the uncertainty part of the evaluation.

## What I would change next

* a defensive book that does not rely on Treasuries when they move with equities
* confidence that falls when signals are stale or contradictory, with escalation below the 0.40 threshold
* a backtest on the 2018 to 2026 window before anything else, since it decided the result
* removing the unused helper methods (`_named_regime`, `_zscore` and others), leftovers from an earlier regime model
