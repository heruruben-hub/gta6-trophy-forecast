# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/heruruben-hub/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/heruruben-hub/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic forecaster for the make.fun bounty **"Forecast how many GTA VI
players earn each trophy in week one"** (gta6-trophy-rarity-forecast-01). Given
every read the house recorded of Exophase's Grand Theft Auto VI (PS5) trophy page
up to a checkpoint, it writes, for each of the game's 55 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of that trophy's earn rate — the percent of tracked
players who will hold it — at the case's target, a week after launch. Scoring is
mean pinball loss on the logit scale, so an honest median and an honest spread
matter equally, and rare trophies count as much as common ones.

## Why rates move: the model

An earn rate is a ratio, and this entry models the two sides of it separately:

    rate(t) = earners(t) / players(t)

In week one the denominator explodes — every Exophase refresh adds thousands of
barely-started players — while each trophy's numerator grows at its own pace, so
most rates *fall* even as more people earn the trophy. That dilution is the main
effect to forecast, and a persistence forecast (the house baseline) cannot see it.

Concretely, `main.py`:

- **Dedupes the history into distinct snapshots.** Exophase refreshes on its own
  schedule (launch night's page sat frozen at 638 players for over four hours,
  then jumped to 3,868), so consecutive identical reads carry no information.
- **Works on the clock x = ln(1 + hours since 2026-11-19T00:00:00Z)**, which
  stretches launch night and compresses the 4.5-day extrapolation to the target.
- **Fits per-trophy log-earner growth** (earners = rate × players tracked) by
  least squares, shrinking each slope toward its tier's and the whole list's
  pooled slopes, so a trophy held by four players cannot swing wildly; fits the
  players-tracked trend on the same clock (whole window blended with the recent
  half); then divides the two extrapolations, each anchored at its last observed
  value.
- **Blends three views** — that earners/players decomposition, a direct
  logit-of-rate trend fit over every distinct snapshot (with the same shrinkage),
  and plain persistence — with weights set by how many distinct snapshots were
  seen (0.55/0.30/0.15 once six or more refreshes are on record).
- **Prices its uncertainty honestly**: per-trophy fit residuals, slope
  uncertainty over the extrapolation gap, a random-walk term in the horizon
  (σ = 0.5·√days, at 80% weight), and a refresh term that fades as more refreshes
  are observed. The final's spread lands near σ ≈ 0.9–1.1 logit units.
- **Handles a page never seen to refresh** (a single distinct snapshot — exactly
  the practice cases' shape): the median shifts down by 0.8 logit units in
  expectation of a diluting refresh and the spread widens to σ = 1.05, instead of
  the baseline's narrow 90-minute band that the first refresh blew through.
- Falls back to tier priors for trophies never observed, tolerates null rates,
  short rate lists and empty histories, and never crashes: an invalid case ranks
  below every valid one, so validity is defended first.

Everything is closed-form least squares on rates clamped to [0.01, 99.99] exactly
as the scorer clamps them — no randomness, no clock reads — so the two grading
runs write byte-identical forecasts.

## Run it

Python 3.13, standard library only, no network. The program reads every `*.in`
case file in the input directory, in name order, and writes one `<id>.out` per
case to the output directory:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

With the environment variables unset it reads `/input` and writes `/output`,
which is how the grader runs it (`python3 -P -s /bot/main.py`, one case per run,
8-second cap — a full launch-window history forecasts in well under a second).
Each output is a single JSON object,
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per
trophy, quantiles non-decreasing, each from 0 to 100.

To try it on the kit's practice cases (whose histories are the flat launch-night
page, checkpoints 03:00 and 03:30 UTC on 2026-11-19, targets 90 minutes later):

```sh
cp /path/to/kit/input/p00.in /path/to/kit/input/p01.in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# out/p00.out and out/p01.out: medians shifted below the last rates with a wide
# spread -- the stance that survives the refresh both cases straddle.
```

## Layout

- `main.py` — the whole entry: case parsing, snapshot dedupe, the shrunk
  per-trophy fits, the blend, and the output writer.
- `LICENSE` — MIT, as the kit's starter ships it.

---

Entered on [make.fun](https://make.fun/?ref=kh6903y0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
