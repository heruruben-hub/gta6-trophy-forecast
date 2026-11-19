"""gta6-trophy-rarity-forecast-01: a mechanistic forecaster for GTA VI trophy earn rates.

The starter (the house baseline) says every rate stays at its last read and adds a
random-walk spread. This entry instead models *why* a rate moves. A rate is a ratio:

    rate(t) = earners(t) / players(t)

where players(t) is how many players Exophase tracks for the game and earners(t) how
many of them hold the trophy. In week one the denominator explodes (every refresh adds
thousands of barely-started players) while the numerator grows trophy by trophy, so
most rates fall even as more people earn the trophy: dilution. The forecaster below

  - collapses the history to its distinct snapshots (Exophase refreshes on its own
    schedule, so consecutive identical reads carry no new information),
  - fits, per trophy, the growth of log earners against x = ln(1 + hours since
    2026-11-19T00:00:00Z) -- a clock that stretches launch night and compresses the
    extrapolation to the target -- shrinking each slope toward its tier's and the
    whole list's pooled slopes so a trophy with four earners cannot swing wildly,
  - fits the growth of players tracked on the same clock and divides the two
    anchored extrapolations for a decomposition forecast,
  - blends that with a direct logit-of-rate trend fit (the house curve baseline's
    idea, but over every distinct snapshot and with the same shrinkage) and with
    plain persistence, the weights set by how many distinct snapshots were seen,
  - and prices its own uncertainty honestly: fit residuals, slope uncertainty over
    the extrapolation gap, a random-walk term in the horizon, and a refresh term
    that is large while few refreshes have been observed. A history with a single
    distinct snapshot has never shown a refresh, so it gets a dilution-shifted
    median and a wide spread instead of the baseline's narrow band.

Everything is closed-form least squares on clamped rates: no randomness, no clock
reads, no iteration order dependence, so two runs write byte-identical files.

Run contract (the grader's): `python3 -P -s /bot/main.py`, Python 3.13, standard
library only, no network. It reads every *.in in $MAKEFUN_INPUT_DIR (default
/input) in name order and writes <case id>.out to $MAKEFUN_OUTPUT_DIR (default
/output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put this file's directory back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each scored level (the values the rules name).
Z_KNOWN = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
# The curve baseline's clock: x = ln(1 + hours since this instant).
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# The scorer clamps a percent to this range before its logit; mirror it exactly.
RATE_LO, RATE_HI = 0.01, 99.99
# Fallback medians (percent) for a trophy never seen in any read, by tier.
TIER_PRIOR_RATE = {"platinum": 0.05, "gold": 0.4, "silver": 1.2, "bronze": 2.5}
# A history that shows one distinct snapshot has never shown a refresh: expect one,
# diluting the median by this many logit units, with this spread (see forecast_flat).
FLAT_SHIFT = -0.8
FLAT_SIGMA = 1.05


def clamp_rate(pct: float) -> float:
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up; whole numbers come back as int (100, not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def z_at(level: float) -> float:
    if level in Z_KNOWN:
        return Z_KNOWN[level]
    # Any other level: invert the normal CDF by bisection on the error function.
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clock_x(moment: datetime) -> float:
    """ln(1 + hours since the epoch), never negative."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def ols(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares y ~ x over (x, y) points: (slope, residual sigma, Sxx).

    With fewer than two distinct x values there is no slope to fit: slope 0.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0.0:
        return 0.0, 0.0, 0.0
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    if n > 2:
        resid = sum((p[1] - (my + slope * (p[0] - mx))) ** 2 for p in points)
        sigma = math.sqrt(resid / (n - 2))
    else:
        sigma = 0.0
    return slope, sigma, sxx


def shrink(slope: float, sxx: float, tier_slope: float, global_slope: float) -> float:
    """Pull a per-trophy slope toward its tier's, then the whole list's, pooled slope.

    The weights are in Sxx units: a trophy whose own points span the clock widely
    keeps its slope; a trophy seen at one or two nearby moments mostly inherits.
    """
    w_tier, w_global = 1.5, 0.75
    total = sxx + w_tier + w_global
    return (sxx * slope + w_tier * tier_slope + w_global * global_slope) / total


def dedupe(history: list[dict]) -> list[dict]:
    """Collapse consecutive reads that show the same page into one snapshot each.

    A snapshot keeps the LAST read's time for its moment (the page it shows was
    still current then) and the shared players count and rates.
    """
    snaps: list[dict] = []
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if snaps and snaps[-1]["key"] == key:
            snaps[-1]["at"] = read.get("at")
        else:
            snaps.append(
                {
                    "key": key,
                    "at": read.get("at"),
                    "players": read.get("players_tracked"),
                    "rates": rates,
                }
            )
    for snap in snaps:
        snap["x"] = clock_x(parse_time(snap["at"])) if snap["at"] else 0.0
    return snaps


def pooled_slope(series: list[tuple[float, float, float]]) -> float:
    """Sxx-weighted mean of per-trophy (slope, sigma, sxx) fits; 0 when empty."""
    num = sum(slope * sxx for slope, _sig, sxx in series if sxx > 0.0)
    den = sum(sxx for _slope, _sig, sxx in series if sxx > 0.0)
    return num / den if den > 0.0 else 0.0


def forecast_flat(last_rate: float | None, tier: str, horizon_minutes: float) -> tuple[float, float]:
    """Median logit and sigma for a trophy whose page was never seen to refresh.

    One distinct snapshot means every read showed the same page. Exophase refreshes
    on its own schedule, hours apart, and a refresh in week one mostly adds new,
    barely-started players: rates dilute. So the median sits below the last rate by
    a shift that is full once the horizon reaches an hour, and the spread is wide
    enough to cover the jump landing anywhere in the window.
    """
    if last_rate is None:
        last_rate = TIER_PRIOR_RATE.get(tier, 1.0)
    scale = min(1.0, max(horizon_minutes, 0.0) / 60.0)
    return logit(last_rate) + FLAT_SHIFT * scale, FLAT_SIGMA


def forecast_case(case: dict) -> dict:
    trophies = case.get("trophies") or []
    levels = case.get("levels") or LEVELS
    zs = [z_at(float(tau)) for tau in levels]
    horizon_minutes = float(case.get("horizon_minutes") or 0.0)
    days = horizon_minutes / 1440.0
    target_x = clock_x(parse_time(case["target"])) if case.get("target") else 0.0

    snaps = dedupe(case.get("history") or [])
    distinct = len(snaps)

    # The players-tracked trend, anchored at the last snapshot: players_pred is the
    # denominator every decomposition forecast divides by. The slope blends the
    # whole window's fit with the recent half's: launch-night growth is explosive
    # and bends, and the recent half says which way it is bending now.
    players_pred: float | None = None
    last_snap = snaps[-1] if snaps else None
    if last_snap is not None and last_snap["players"]:
        p_points = [(s["x"], float(s["players"])) for s in snaps if s["players"]]
        p_slope, _p_sig, _p_sxx = ols(p_points)
        if len(p_points) >= 4:
            recent_slope, _r_sig, _r_sxx = ols(p_points[len(p_points) // 2 :])
            p_slope = 0.5 * p_slope + 0.5 * recent_slope
        p_slope = max(p_slope, 0.0)  # tracked players do not leave in week one
        gap = max(target_x - last_snap["x"], 0.0)
        players_pred = float(last_snap["players"]) + p_slope * gap
        players_pred = min(players_pred, float(last_snap["players"]) * 4.0)
        players_pred = max(players_pred, float(last_snap["players"]))

    # Per-trophy series over the distinct snapshots: logit of rate, and log earners
    # (earners = rate% of the players that read tracked, +1 so a zero count fits).
    per_trophy: list[dict] = []
    for i, trophy in enumerate(trophies):
        logit_pts: list[tuple[float, float]] = []
        earn_pts: list[tuple[float, float]] = []
        last_rate: float | None = None
        last_x = 0.0
        last_earners = 0.0
        for snap in snaps:
            rates = snap["rates"]
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            rate = float(rate)
            logit_pts.append((snap["x"], logit(rate)))
            if snap["players"]:
                earners = rate / 100.0 * float(snap["players"])
                earn_pts.append((snap["x"], math.log(earners + 1.0)))
                last_earners = earners
            last_rate = rate
            last_x = snap["x"]
        per_trophy.append(
            {
                "tier": trophy.get("tier", "bronze"),
                "logit_fit": ols(logit_pts),
                "earn_fit": ols(earn_pts),
                "last_rate": last_rate,
                "last_x": last_x,
                "last_earners": last_earners,
                "has_earn": bool(earn_pts),
            }
        )

    # Pooled slopes per tier and overall, for each of the two fits: the shrinkage
    # targets that keep a noisy trophy near what trophies like it are doing.
    def pool(key: str, tier: str | None) -> float:
        return pooled_slope(
            [pt[key] for pt in per_trophy if tier is None or pt["tier"] == tier]
        )

    tiers = {pt["tier"] for pt in per_trophy}
    logit_tier = {t: pool("logit_fit", t) for t in tiers}
    earn_tier = {t: pool("earn_fit", t) for t in tiers}
    logit_global = pool("logit_fit", None)
    earn_global = pool("earn_fit", None)

    # How much to trust each view of the future, by how many refreshes were seen.
    if distinct >= 6:
        w_decomp, w_direct, w_last = 0.55, 0.30, 0.15
    elif distinct >= 3:
        w_decomp, w_direct, w_last = 0.40, 0.30, 0.30
    elif distinct == 2:
        w_decomp, w_direct, w_last = 0.30, 0.25, 0.45
    else:
        w_decomp, w_direct, w_last = 0.0, 0.0, 1.0

    # Uncertainty that has nothing to do with any one trophy's fit: a random walk
    # over the horizon, plus a refresh term that fades as refreshes are observed.
    rw_var = (0.5 * math.sqrt(days)) ** 2
    refresh_var = 0.55**2 * math.exp(-max(distinct - 1, 0) / 3.0)

    rows = []
    for trophy, pt in zip(trophies, per_trophy):
        tier = pt["tier"]
        if pt["last_rate"] is None or distinct <= 1:
            median, sigma = forecast_flat(pt["last_rate"], tier, horizon_minutes)
        else:
            gap = max(target_x - pt["last_x"], 0.0)
            lg_slope, lg_resid, lg_sxx = pt["logit_fit"]
            lg_slope = shrink(lg_slope, lg_sxx, logit_tier[tier], logit_global)
            logit_direct = logit(pt["last_rate"]) + lg_slope * gap

            logit_decomp: float | None = None
            if pt["has_earn"] and players_pred:
                e_slope, _e_sig, e_sxx = pt["earn_fit"]
                e_slope = shrink(e_slope, e_sxx, earn_tier[tier], earn_global)
                e_pred = math.exp(math.log(pt["last_earners"] + 1.0) + e_slope * gap) - 1.0
                e_pred = min(max(e_pred, 0.8 * pt["last_earners"]), players_pred)
                logit_decomp = logit(100.0 * e_pred / players_pred)

            median = w_direct * logit_direct + w_last * logit(pt["last_rate"])
            w_sum = w_direct + w_last
            if logit_decomp is not None:
                median += w_decomp * logit_decomp
                w_sum += w_decomp
            median /= w_sum

            # The spread: half the fit's residual variance (the blend averages two
            # views, which cancels some noise), the random walk at 80% weight, the
            # slope's own uncertainty over the gap, and the refresh term.
            resid = min(max(lg_resid, 0.15), 1.2) if distinct >= 3 else 0.35
            slope_err = (resid / math.sqrt(lg_sxx) * gap) if lg_sxx > 0.0 else 0.0
            var = 0.5 * resid**2 + 0.64 * rw_var + slope_err**2 + refresh_var
            sigma = min(max(math.sqrt(var), 0.30), 1.75)
            if days >= 3.0:
                sigma = max(sigma, 0.85)

        quantiles: list[float | int] = []
        running = 0.0
        for z in zs:
            value = inv_logit(median + z * sigma)
            value = min(max(value, 0.0), 100.0)
            running = max(running, value)  # keep the row non-decreasing
            quantiles.append(round4(running))
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_name = f"{case.get('id') or path.stem}.out"
        (OUTPUT_DIR / out_name).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
