# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/van6119918/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/van6119918/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at the case's target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) in percent. One file, `main.py`, Python 3.13 standard library only — no network, no packages, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing, 0–100.

## The model — earner counts, not rates

A trophy's rate is earners / players-tracked, and the two move for different reasons: earners grow as players play, players-tracked grows as more profiles are scanned (and Exophase refreshes its counts in bursts, so the displayed page can sit unchanged for hours and then jump). Fitting the rate directly conflates the two, so the program decomposes every read into `N(t)` (players tracked) and `E_i(t) = rate_i(t) * N(t)` (earners of trophy i) and fits each on a log-log scale in hours since release:

```
ln N   = a  + alpha  * ln(1 + t)
ln E_i = c_i + beta_i * ln(1 + t)
```

then forecasts N and E_i at the target and divides. The log-log form makes both growth processes decelerate, which is what a launch surge followed by a long tail looks like. A trophy whose earners grow faster than the tracked population rises; one whose earners lag falls.

## The stale-page / refresh prior

On launch night the page sat unchanged for hours (638 players tracked, every rate the same from 23:48 UTC to 04:00 UTC on 2026-11-19) and then refreshed to 3,868 players with every rate but the 100% one lower (per the competition rules). A case whose history is flat therefore says nothing about trends — but it does say a refresh is due. When the fitted slopes are unidentified (flat history), the program falls back to a refresh prior:

- **N jumps by the power law with alpha = 0.95**, calibrated on that first observed refresh: 638 → 3,868 players as (1 + t) went 0.83 → 5.5 hours past release, i.e. ln(3868/638) / ln(5.5/0.83) = 0.95. (An alpha of 1.0 would give 4,230; 0.95 gives 3,843, close to the observed 3,868.)
- **New joiners earn trophy i at a fraction kappa of the incumbents' rate.** Kappa is read off the trophy's description: the intro trophy everyone already has stays at ~100% (kappa 1.0); early story chapters are earned by a good share of fresh players (0.35 for Chapter 1, falling down the story); grind trophies almost nobody new has (0.02). Every kappa is an **estimate**, and the rules' own statement that every non-100% rate fell at the refresh is what pins kappa below 1 for every other trophy.
- **Incumbents keep playing**, so their earners grow a little (5% for story trophies, 2% otherwise) over the horizon.

The two paths are blended in log space by how much the history actually moves: weight `w = min(1, (distinct_states - 1) / 4)`. A flat history (the practice cases) is all prior; a history with five or more distinct states (the final, days of reads) is all trend fit. Consecutive identical reads collapse to one state, and a null rate never opens a new state on its own.

## Spreads

Quantiles are logit-normal around the median: `q_tau = logistic(logit(m) + z_tau * s)`, `z = (-1.645, -0.674, 0, 0.674, 1.645)`. The spread `s` combines a horizon term (`0.30 * sqrt(days)`), the per-trophy fit residual (clamped 0.1–0.6, widened when the target lies far past the data), and a refresh term (`0.45 * (1 - w)`) that is full while the forecast rests on the refresh prior and fades as real trend data takes over.

## Validation (local, no true practice answers exist)

- **Practice cases, assumed truths:** assuming the 04:30 refresh multiplies N by 6.06 and new joiners earn each trophy at a fraction `k` of the incumbents' rate, this program's mean pinball loss beats the house baseline's for every `k` from 0 to 0.35 (e.g. at `k = 0.1`: 0.13 vs 1.28; at `k = 0.2`: 0.16 vs 1.00). The house's own stated practice loss for its baseline, ~1.51, sits near `k ≈ 0.04` in this family — where this program scores ~0.15–0.26.
- **Synthetic final:** a fake 3-day case with power-law N/E growth and refresh jumps, scored against its known truth: this program 0.214 vs the house baseline 0.243, with a typical median miss of 0.06 vs 0.77 percentage points.
- **Format/determinism:** outputs are valid (55 rows, non-decreasing quantiles, 0–100), byte-identical across runs, ~0.07 s per case — far inside the 8 s limit.

## Assumptions and estimates

Everything labelled **estimate** above (the 0.95 launch alpha, the kappa table, the 0.30 horizon volatility, the 0.45 refresh uncertainty, the 5%/2% incumbent progression) is the author's judgement, calibrated on the one refresh the rules describe — not measured data, and not presented as such. No outside data is used at run time: everything comes from the case file, so the program runs in the grader's no-network sandbox.

---

<sub>Entered on [make.fun](https://make.fun/?ref=eaa6wt68&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
