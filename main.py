"""gta6-trophy-rarity-forecast-01 entry: power-law extrapolation of earners and players tracked.

Model (written for this entry):
  - Time origin 2026-11-19T00:00:00Z; for a read at time t, h = hours since origin
    (clamped to >= -0.9) and x = ln(1 + h).
  - History is deduped to distinct snapshots: consecutive reads with identical
    players_tracked AND identical rates collapse to one (Exophase refreshes
    discretely, so plateaus carry no new information).
  - log(players_tracked) = a_N + b_N * x, fit by OLS over distinct snapshots.
  - Per trophy, implied earners E = rate * N / 100 per snapshot (rate in percent;
    null rates skipped; rate 0 uses E = 0.5). log(E) = a + b * x, fit by OLS.
  - Tier shrinkage: each trophy's slope is shrunk toward its tier's median slope,
    w = m / (m + 4) with m the number of snapshots used for that trophy; the
    shrunk slope is capped to [-1, 4].
  - Forecast at x* = ln(1 + h_target): N* and E* from the fitted lines; the
    median rate is 100 * E* / N*, clamped to [0.01, 99.99].
  - Spread (logit units) combines the per-trophy residual std (shrunk halfway
    toward its tier's median residual std, floored at 0.10, capped at 1.2), a
    random-walk term 0.5 * sqrt(horizon_days), and an extrapolation term
    0.25 * max(0, x* - x_last). With at most 2 distinct snapshots (no refresh
    observed yet) an extra 0.9 is added in quadrature: an unseen Exophase
    refresh can dilute rates sharply.
  - Quantiles from the median on the logit scale with z = [-1.645, -0.674, 0,
    0.674, 1.645], rounded to 4 decimals half up.

Python 3.13, standard library only, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Time origin: GTA VI release day, 2026-11-19T00:00:00Z.
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Spread pieces (logit units).
RW_SIGMA = 0.5          # random-walk spread per sqrt(day)
EXT_COEF = 0.25         # extrapolation penalty per unit of log-time beyond the last read
NO_REFRESH_EXTRA = 0.9  # added in quadrature when <= 2 distinct snapshots exist
S_FIT_FLOOR = 0.10
S_FIT_CAP = 1.2
MEDIAN_FLOOR = 0.01
MEDIAN_CAP = 99.99
SLOPE_CAP = 4.0
SHRINK_K = 4.0          # w = m / (m + 4)


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp (with 'Z' or an offset)."""
    if text is None:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def log_time(dt):
    """x = ln(1 + h) for h = hours since ORIGIN, h clamped to >= -0.9."""
    h = (dt - ORIGIN).total_seconds() / 3600.0
    if h < -0.9:
        h = -0.9
    return math.log1p(h)


def clamp_pct(x):
    return min(MEDIAN_CAP, max(MEDIAN_FLOOR, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    # Numerically safe logistic back to percent.
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (like the starter's round4)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def ols(xs, ys):
    """Ordinary least squares y = a + b*x.

    Returns (a, b, resid_std). With fewer than 2 points, or no spread in x,
    the slope is 0 and the intercept is the mean of y. With fewer than 3
    points the residual std is 0.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_y = sum(ys) / n
    if n < 2:
        return mean_y, 0.0, 0.0
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    if n < 3:
        return a, b, 0.0
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, math.sqrt(sse / (n - 2))


def dedupe(history):
    """Collapse consecutive reads with identical players_tracked and rates."""
    snaps = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(read)
    return snaps


def forecast_case(case, name):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    target = parse_time(case.get("target"))
    if target is None:
        at = parse_time(case.get("at"))
        if at is not None and case.get("horizon_minutes"):
            from datetime import timedelta

            target = at + timedelta(minutes=case["horizon_minutes"])
    if target is None:
        target = ORIGIN
    x_star = log_time(target)

    # Empty history: no information at all.
    if not history:
        spread = 2.0
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append(
                {
                    "id": trophy["id"],
                    "q": [round4(min(100.0, max(0.0, inv_logit(lg + z * spread)))) for z in Z],
                }
            )
        return {"trophies": rows}

    snaps = dedupe(history)
    n_snaps = len(snaps)

    # Snapshot log-times (a read with an unparseable time falls back to the
    # previous snapshot's x, or 0 for the first).
    xs = []
    last_x = 0.0
    for read in snaps:
        dt = parse_time(read.get("at"))
        x = log_time(dt) if dt is not None else last_x
        xs.append(x)
        last_x = x
    x_last = xs[-1] if xs else 0.0

    # Fit log(players_tracked) against x.
    n_vals = []
    for read in snaps:
        n = read.get("players_tracked")
        n_vals.append(math.log(n) if isinstance(n, (int, float)) and n > 0 else None)
    n_pairs = [(x, y) for x, y in zip(xs, n_vals) if y is not None]
    if n_pairs:
        a_n, b_n, _ = ols([p[0] for p in n_pairs], [p[1] for p in n_pairs])
    else:
        a_n, b_n = 0.0, 0.0
    n_star = math.exp(a_n + b_n * x_star)
    if not (n_star > 0) or not math.isfinite(n_star):
        n_star = 1.0

    # Last non-null rate per trophy (for the no-data fallback).
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(snaps):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        last_rates.append(rate if rate is not None else MEDIAN_FLOOR)

    # Per-trophy fits of log(earners) against x.
    fits = []
    for i in range(len(trophies)):
        pts_x, pts_y = [], []
        for x, read in zip(xs, snaps):
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            n = read.get("players_tracked")
            if not isinstance(n, (int, float)) or n <= 0:
                continue
            earners = rate * n / 100.0
            if earners <= 0:
                earners = 0.5
            pts_x.append(x)
            pts_y.append(math.log(earners))
        if pts_x:
            a_i, b_i, s_i = ols(pts_x, pts_y)
            m = len(pts_x)
        else:
            # No usable rate at all: anchor on the fallback rate and let its
            # earners grow with players tracked, so the median rate stays at
            # the last (fallback) rate instead of diluting as N* grows.
            n_last = snaps[-1].get("players_tracked") or 1
            earners = last_rates[i] * n_last / 100.0
            if earners <= 0:
                earners = 0.5
            a_i, b_i, s_i = math.log(earners) - b_n * x_last, b_n, 0.0
            m = 0
        fits.append({"a": a_i, "b": b_i, "s": s_i, "m": m, "tier": trophies[i].get("tier")})

    # Tier medians of slope and residual std, over trophies that actually have
    # a fit (m > 0); a no-data trophy must not drag its tier's medians.
    tier_slopes = {}
    tier_resid = {}
    for tier in {f["tier"] for f in fits}:
        members = [f for f in fits if f["tier"] == tier and f["m"] > 0]
        tier_slopes[tier] = median([f["b"] for f in members]) if members else 0.0
        tier_resid[tier] = median([f["s"] for f in members]) if members else 0.0

    s_rw = RW_SIGMA * math.sqrt(max(0.0, horizon_days))
    s_ext = EXT_COEF * max(0.0, x_star - x_last)
    flat_extra = NO_REFRESH_EXTRA if n_snaps <= 2 else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        tier = f["tier"]
        b_tier = tier_slopes.get(tier, 0.0)
        if f["m"] > 0:
            w = f["m"] / (f["m"] + SHRINK_K)
            b_star = w * f["b"] + (1.0 - w) * b_tier
        else:
            # No data for this trophy: keep its anchor slope (earners grow
            # with players tracked), so the median stays at the last rate.
            b_star = f["b"]
        b_star = min(SLOPE_CAP, max(-SLOPE_CAP, b_star))

        e_star = math.exp(f["a"] + b_star * x_star)
        if not math.isfinite(e_star) or e_star <= 0:
            e_star = 0.5
        med = clamp_pct(100.0 * e_star / n_star)

        s_fit = 0.5 * f["s"] + 0.5 * tier_resid.get(tier, 0.0)
        s_fit = min(S_FIT_CAP, max(S_FIT_FLOOR, s_fit))
        spread = math.sqrt(s_fit * s_fit + s_rw * s_rw + s_ext * s_ext + flat_extra * flat_extra)

        lg = logit(med)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(min(100.0, max(0.0, inv_logit(lg + z * spread))))
            # Rounding can only tie, never decrease, but clamp defensively.
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash on a case: fall back to an empty-history forecast.
            trophies = []
            try:
                trophies = json.loads(path.read_text(encoding="utf-8")).get("trophies") or []
            except Exception:
                trophies = []
            rows = []
            for trophy in trophies:
                lg = logit(1.0)
                rows.append(
                    {
                        "id": trophy["id"],
                        "q": [round4(min(100.0, max(0.0, inv_logit(lg + z * 2.0)))) for z in Z],
                    }
                )
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
