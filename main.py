"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at the case's target, as five quantiles (0.05/0.25/0.5/0.75/
0.95) in percent. Python 3.13, standard library only, no network, deterministic.

Model — earner counts, not rates
-------------------------------
A trophy's rate is earners / players-tracked, and the two move for different
reasons: earners grow as players play, players-tracked grows as more profiles
are scanned (and Exophase refreshes its counts in bursts, so the displayed
page can sit unchanged for hours and then jump). Fitting the rate directly
conflates the two, so the program decomposes every read:

    N(t)   = players tracked
    E_i(t) = rate_i(t) * N(t)   (earners of trophy i)

and fits each on a log-log scale in hours since release:

    ln N   = a  + alpha  * ln(1 + t)
    ln E_i = c_i + beta_i * ln(1 + t)

then forecasts N and E_i at the target and divides. The log-log form makes
both growth processes decelerate (a fixed proportional gain per doubling of
age), which is what a launch surge followed by a long tail looks like; the
rate's own log-log slope is beta_i - alpha, so a trophy whose earners grow
faster than the tracked population rises, and one whose earners lag falls.

The stale-page / refresh prior
------------------------------
On launch night the page sat unchanged for hours (638 players tracked, every
rate the same from 23:48 UTC to 04:00 UTC on 2026-11-19) and then refreshed
to 3,868 players with every rate but the 100% one lower (RULES.txt). A case
whose history is flat therefore says nothing about trends — but it does say a
refresh is due. When the fitted slopes are unidentified (flat history), the
program falls back to a refresh prior:

  * N jumps by the power law with alpha = 0.95, calibrated on that first
    observed refresh: 638 -> 3,868 players as (1 + t) went 0.83 -> 5.5 hours
    past release, i.e. ln(3868/638) / ln(5.5/0.83) = 0.95. (An alpha of 1.0
    would give 4,230; 0.95 gives 3,843, close to the observed 3,868.)
  * New joiners earn trophy i at a fraction kappa_i of the incumbents' rate.
    kappa is read off the trophy's description (see kappa_for): the intro
    trophy everyone already has stays at ~100% (kappa 1.0); early story
    chapters are earned by a good share of fresh players (0.35 for Chapter 1,
    falling down the story); grind trophies almost nobody new has (0.02).
    Every kappa is an ESTIMATE, labelled as such; the rules' own statement
    that every non-100% rate fell at the refresh is what pins kappa below 1
    for every other trophy.
  * Incumbents keep playing, so their earners grow a little (5% for story
    trophies, 2% otherwise) over the horizon.

The two paths are blended in log space by how much the history actually
moves: weight w = min(1, (distinct_states - 1) / 4). A flat history (the
practice cases) is all prior; a history with five or more distinct states
(the final, days of reads) is all trend fit.

Spreads (quantiles)
-------------------
Quantiles are logit-normal around the median: q_tau = logistic(logit(m) +
z_tau * s), z = (-1.645, -0.674, 0, 0.674, 1.645). The spread s combines
  * a horizon term, 0.30 * sqrt(days), the irreducible drift over the horizon;
  * the per-trophy fit residual (clamped 0.1..0.6), widened when the target
    lies far past the data;
  * a refresh term, 0.45 * (1 - w): full when the forecast rests on the
    refresh prior, fading as real trend data takes over.
The house baseline's spread is 0.5 * sqrt(days) around a no-change median and
scores about 1.51 on the practice, because the refresh moves every rate
outside its narrow 90-minute band; this program's edge is a median that
moves with the refresh and the trends, not a narrower spread.

Assumptions and estimates (all labelled in the README, none presented as
measured data): the 0.95 launch alpha, the kappa table, the 0.30 horizon
volatility, the 0.45 refresh uncertainty. No outside data is used at run
time: everything comes from the case file, so the program runs in the
grader's no-network sandbox.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path; nothing beside this file is
# imported, but keep the line harmless for parity with the starter.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-night refresh calibration (see module docstring): the power-law
# exponent of players-tracked growth, and the cap on the jump it implies.
ALPHA_PRIOR = 0.95
# Horizon volatility of a rate's logit, per square-root day.
SIGMA_DAY = 0.30
# Extra logit spread while the forecast rests on the refresh prior.
SIGMA_REFRESH = 0.45
# Release: midnight UTC on the game's release date (the house curve's origin).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return (when - RELEASE).total_seconds() / 3600.0


def log_time(when: datetime) -> float:
    """ln(1 + hours since release), floored so pre-release reads stay finite.

    The first reads are minutes before midnight UTC on release day, so
    1 + t is 0.83 there, not 1: clamping t at 0 would understate the
    launch-night growth (the observed 638 -> 3,868 jump is a 6.02x power-law
    step from 0.83 to 5.5, not 5.05x from 1 to 5.5). Floor 1 + t at 0.5.
    """
    return math.log(max(0.5, 1.0 + hours_since_release(when)))


def linfit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return (a, b, resid)


def kappa_for(trophy: dict, last_rate: float) -> float:
    """New-joiner earn fraction for a trophy (ESTIMATE — see docstring).

    A player who has just been scanned onto Exophase has played far less
    than the incumbents, so they earn a trophy at a fraction kappa of the
    incumbents' rate. The fraction is read off what the trophy asks for.
    """
    if last_rate is not None and last_rate >= 99.0:
        return 1.0
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "bronze")
    if "gone fishing" in text:
        return 1.0
    if "chapter 1" in text:
        return 0.35
    if "chapter 2" in text:
        return 0.25
    if "chapter 3" in text:
        return 0.20
    if "chapter 4" in text:
        return 0.15
    if "100% completion" in text or "250 stunts" in text or "100 locations" in text \
            or "50 challenges" in text or "collectible card sets" in text \
            or "gold medal in all" in text:
        return 0.02
    if "final mission for" in text:
        return 0.10
    if "final mission" in text:
        return 0.08
    if "laundering" in text or "garage properties" in text:
        return 0.03
    if "try each type of activity" in text or "arcade game" in text \
            or "cocktail" in text or "bowling" in text:
        return 0.25
    if "meditate at every spot" in text:
        return 0.05
    # Tier fallback: rarer tiers take longer to earn.
    return {"platinum": 0.02, "gold": 0.05, "silver": 0.06, "bronze": 0.12}.get(tier, 0.12)


def progression_for(trophy: dict) -> float:
    """Incumbent earners' fractional growth over a short horizon (ESTIMATE)."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if "chapter" in text or "final mission" in text:
        return 0.05
    return 0.02


def _rates_differ(a, b):
    """True if two rate lists differ where both are known.

    A null is a missing value, not a change: a read that merely lacks a
    trophy must not count as a new page state. A real change is a position
    where both reads have a rate and the rates differ.
    """
    for x, y in zip(a, b):
        if x is not None and y is not None and x != y:
            return True
    return False


def distinct_states(history):
    """Collapse consecutive identical reads to one snapshot each.

    Exophase's page sits unchanged between refreshes, so a run of identical
    reads is one state; keeping them all would overweight the stale period
    in the fits. A state changes when the players-tracked count changes or
    a known rate changes; nulls alone never open a new state. Returns
    [(time, players, rates)] in time order.
    """
    states = []
    prev_n, prev_rates = None, None
    for read in history:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if prev_n is not None and n == prev_n and not _rates_differ(prev_rates, rates):
            # Same state; fill in any nulls this read now knows, so a null
            # in the reference cannot mask a real change at that position
            # in a later read.
            merged = list(prev_rates)
            for i, v in enumerate(rates):
                if v is not None and i < len(merged):
                    merged[i] = v
            prev_rates = merged
            continue
        prev_n, prev_rates = n, rates
        states.append((parse_time(read["at"]), n, rates))
    return states


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_time(target)

    states = distinct_states(history)
    n_states = len(states)
    # How much the history actually moves: 0 for a flat page, 1 once five
    # or more distinct states pin the trends down.
    w = min(1.0, max(0.0, (n_states - 1) / 4.0))

    # Last non-null rate per trophy, and the last change time of the page.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for _, _, rates in reversed(states):
            if i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        last_rates.append(rate if rate is not None else 0.0)
    last_change_at = states[-1][0]
    last_n = states[-1][1] or 0

    # --- Players-tracked trend: ln N = a + alpha * ln(1 + t). ---
    xs_n, ys_n = [], []
    for when, n, _ in states:
        if n and n > 0:
            xs_n.append(log_time(when))
            ys_n.append(math.log(n))
    fit_n = linfit(xs_n, ys_n)
    if fit_n is not None and len(set(xs_n)) > 1:
        _, alpha_fit, _ = fit_n
        alpha = alpha_fit
    else:
        alpha = ALPHA_PRIOR
    # N at the target, from the last change time (a stale displayed N is
    # the pre-refresh value; the power law carries it to the target).
    x_last_change = log_time(last_change_at)
    if last_n > 0:
        n_target = last_n * math.exp(alpha * max(0.0, x_target - x_last_change))
    else:
        n_target = 1.0
    n_target = max(n_target, last_n if last_n > 0 else 1.0)

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        e_last = max(r_last / 100.0 * last_n, 0.0) if last_n > 0 else 0.0

        # --- Trend path: ln E_i = c + beta * ln(1 + t). ---
        xs_e, ys_e = [], []
        for when, n, rates in states:
            if i < len(rates) and rates[i] is not None and n and n > 0:
                e = rates[i] / 100.0 * n
                xs_e.append(log_time(when))
                ys_e.append(math.log(max(e, 0.5)))
        fit_e = linfit(xs_e, ys_e)
        resid = 0.0
        if fit_e is not None and len(set(xs_e)) > 1:
            a_e, beta_e, resid = fit_e
            e_trend = math.exp(a_e + beta_e * x_target)
        else:
            fit_e = None
            e_trend = None

        # --- Refresh-prior path: incumbents progress a little, and the
        # jump in tracked players brings new joiners at kappa * r_last. ---
        kappa = kappa_for(trophy, r_last)
        prog = progression_for(trophy)
        e_prior = e_last * (1.0 + prog) + max(0.0, n_target - last_n) * kappa * (r_last / 100.0)
        if e_last <= 0.0:
            # Nobody has it yet: half an earner keeps the log finite and the
            # forecast off exactly zero.
            e_prior = max(e_prior, 0.5)

        # --- Blend in log space, then divide by N for the median rate. ---
        if e_trend is not None and e_trend > 0:
            log_e = w * math.log(e_trend) + (1.0 - w) * math.log(max(e_prior, 0.5))
        else:
            log_e = math.log(max(e_prior, 0.5))
        e_target = math.exp(log_e)
        # Earners can never exceed the players tracked.
        e_target = min(e_target, n_target)
        median = clamp_pct(100.0 * e_target / n_target) if n_target > 0 else r_last

        # --- Spread: horizon drift + fit residual + refresh uncertainty. ---
        s_base = SIGMA_DAY * math.sqrt(max(0.0, horizon_days))
        if fit_e is not None:
            x_min, x_max = min(xs_e), max(xs_e)
            span = max(x_max - x_min, 1e-9)
            extrap = max(0.0, x_target - x_max) / span
            s_fit = min(0.6, max(0.1, resid)) * (1.0 + 0.5 * min(2.0, extrap))
        else:
            s_fit = 0.0
        s_refresh = SIGMA_REFRESH * (1.0 - w)
        spread = math.sqrt(s_base ** 2 + s_fit ** 2 + s_refresh ** 2)
        spread = min(spread, 2.5)

        med_logit = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = clamp_pct(logistic(med_logit + z * spread))
            q = round4(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never leave a case without an output file
            print(f"warning: {path.name} failed ({exc}); writing fallback", file=sys.stderr)
            rows = []
            for trophy in case["trophies"]:
                lg = logit(1.0)
                qs = [round4(clamp_pct(logistic(lg + z * 1.0))) for z in Z]
                rows.append({"id": trophy["id"], "q": qs})
            rows.sort(key=lambda row: row["id"])
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
