# makefun-gta6-trophy-rarity-forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gongdewei98-design/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gongdewei98-design/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Model

Each trophy's earn rate is earners / players-tracked, with both quantities extrapolated as power laws in log-time since 2026-11-19T00:00:00Z: `log(N)` and `log(E)` are fit by least squares against `ln(1 + hours)`. History is first deduped to distinct snapshots (consecutive reads with identical players-tracked and rates collapse to one, since Exophase refreshes discretely). Each trophy's slope is shrunk toward its tier's median slope (`w = m / (m + 4)`), capped to [-1, 4]. The median forecast is `100 * E* / N*` at the target, clamped to [0.01, 99.99].

The spread on the logit scale combines the per-trophy residual std (shrunk halfway toward its tier's median, floored at 0.10, capped at 1.2), a random-walk term `0.5 * sqrt(horizon_days)`, and an extrapolation term `0.25 * max(0, x* - x_last)`. When the history holds at most 2 distinct snapshots — no refresh observed yet — an extra 0.9 is added in quadrature, because an unseen Exophase refresh can dilute rates sharply (the first refresh took players tracked from 638 to 3,868 and lowered almost every rate).

Python 3.13, standard library only, no network, fully deterministic.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes `<id>.out` for each to the output directory.

---

Built for [make.fun](https://make.fun/r/5t4wfmsa) — bounty `gta6-trophy-rarity-forecast-01`.
