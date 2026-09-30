"""
Regular, gap-aware, smoothed time series per field, built from output/fields.csv.

Turns irregular satellite passes into one value every --step days, on a grid shared by
every field, which comparisons across fields and years (and a classifier) need:

    1. drop flagged observations (haze, dips, scene haze)
    2. linear interpolation onto the grid, never across a gap longer than --max-gap days
       and never before the first or after the last observation; those stay empty
    3. Savitzky-Golay smoothing, with the window chosen by hold-out validation
    4. phenology per field and crop season from the smoothed NDVI: peak, start, end, length

    python s2_fields.py        # first, to produce output/fields.csv
    python s2_series.py
    python s2_series.py --window 7 --max-gap 20

Writes to ./output/:
    series.csv          one row per field, index and grid date: interpolated and smoothed value
    series_matrix.csv   one row per field, one column per index and grid date (smoothed)
    phenology.csv       one row per field and crop season
    series.png          observations, gaps, smoothed NDVI and season dates per field
"""
import argparse
import os
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from s2_indices import INDICES, shade_seasons

WINDOWS = [5, 7, 9, 11]  # Savitzky-Golay windows tried by the hold-out, in grid steps (5 = 25 days)
COLORS = plt.cm.tab10.colors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join("output", "fields.csv"), help="output of s2_fields.py")
    p.add_argument("--step", type=int, default=5, help="grid spacing in days (default 5, Sentinel-2's revisit)")
    p.add_argument("--start", help="first grid date, YYYY-MM-DD (default: 1 September before the first observation)")
    p.add_argument("--max-gap", type=int, default=30, help="don't interpolate across gaps longer than this (days)")
    p.add_argument("--window", type=int, help="Savitzky-Golay window in grid steps, odd (default: chosen by hold-out)")
    p.add_argument("--order", type=int, default=2, help="Savitzky-Golay polynomial order (default 2)")
    p.add_argument("--min-amplitude", type=float, default=0.2,
                   help="how far NDVI must rise to count as a crop season (default 0.2)")
    p.add_argument("--out", default="output", help="output folder")
    args = p.parse_args()
    if args.window is not None and (args.window % 2 == 0 or args.window <= args.order):
        p.error("--window must be odd and larger than --order")
    return args


def days_since(dates, origin):
    return np.asarray((pd.DatetimeIndex(dates) - origin).days, dtype=float)


def make_grid(obs, args):
    """Grid dates every --step days, from --start (or 1 September) to the last observation."""
    first, last = obs.date.min(), obs.date.max()
    if args.start:
        start = pd.Timestamp(args.start)
    else:
        # The agricultural year here turns around 1 September, between the rice and wheat seasons.
        start = pd.Timestamp(first.year if first.month >= 8 else first.year - 1, 9, 1)
    return pd.date_range(start, last, freq=f"{args.step}D")


def regularise(obs_days, obs_values, grid_days, max_gap):
    """Linear interpolation onto the grid. Empty (NaN) outside the observations and inside long gaps."""
    out = np.interp(grid_days, obs_days, obs_values)
    out[(grid_days < obs_days[0]) | (grid_days > obs_days[-1])] = np.nan
    right = np.clip(np.searchsorted(obs_days, grid_days), 0, len(obs_days) - 1)  # first observation on or after
    left = np.clip(right - 1, 0, len(obs_days) - 1)
    exact = obs_days[right] == grid_days
    out[(obs_days[right] - obs_days[left] > max_gap) & ~exact] = np.nan
    return out


def runs(valid):
    """Index arrays of each unbroken stretch of True."""
    idx = np.flatnonzero(valid)
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []


def smooth(values, window, order):
    """Savitzky-Golay over each unbroken stretch; stretches shorter than the window are left as they are.

    window None means no smoothing, which the hold-out also scores as a baseline.
    """
    out = values.copy()
    if window is None:
        return out
    for run in runs(~np.isnan(values)):
        if len(run) >= window:
            out[run] = savgol_filter(values[run], window, order, mode="interp")
    return out


def value_at(grid_days, series, day):
    """Series value on any day, linear between grid points; NaN next to a gap."""
    i = np.searchsorted(grid_days, day)
    if i < len(grid_days) and grid_days[i] == day:
        return series[i]
    if i == 0 or i == len(grid_days) or np.isnan(series[i - 1]) or np.isnan(series[i]):
        return np.nan
    t = (day - grid_days[i - 1]) / (grid_days[i] - grid_days[i - 1])
    return series[i - 1] + t * (series[i] - series[i - 1])


def holdout(ndvi_obs, grid_days, args, candidates, repeats=20, fraction=0.1, seed=0):
    """Hide a tenth of each field's clear observations, rebuild the series without them, and
    measure how far off it is at the hidden dates. Repeated with different random picks.

    Returns the mean absolute error per window (None = interpolation only) and every error.
    """
    rng = np.random.default_rng(seed)
    errors = {w: [] for w in candidates}
    for _ in range(repeats):
        for field, (days, values) in ndvi_obs.items():
            hide = rng.choice(np.arange(1, len(days) - 1), size=max(1, round(fraction * len(days))), replace=False)
            keep = np.setdiff1d(np.arange(len(days)), hide)
            gridded = regularise(days[keep], values[keep], grid_days, args.max_gap)
            for w in candidates:
                smoothed = smooth(gridded, w, args.order)
                for h in hide:
                    estimate = value_at(grid_days, smoothed, days[h])
                    if not np.isnan(estimate):
                        errors[w].append((field, days[h], abs(estimate - values[h])))
    mae = {w: float(np.mean([e for _, _, e in errs])) for w, errs in errors.items()}
    return mae, errors


def season_windows(first, last):
    """Punjab's crop seasons overlapping the data, named by the year they end in."""
    out = []
    for y in range(first.year - 1, last.year + 1):
        out.append((f"rabi_{y + 1}", pd.Timestamp(y, 11, 1), pd.Timestamp(y + 1, 4, 30)))
        out.append((f"kharif_{y + 1}", pd.Timestamp(y + 1, 5, 1), pd.Timestamp(y + 1, 10, 31)))
    return [s for s in out if s[2] >= first and s[1] <= last]


def crossing(days, values, i_from, i_to, level):
    """Day where the curve, walked from i_from towards i_to, first drops below level."""
    step = 1 if i_to > i_from else -1
    for i in range(i_from, i_to, step):
        a, b = values[i], values[i + step]
        if a >= level > b:
            return days[i] + (a - level) / (a - b) * (days[i + step] - days[i])
    return np.nan


def phenology(field, grid, smoothed, args):
    """Peak, start, end and length of each crop season, from smoothed NDVI.

    Start and end are where the curve is halfway between its base and its peak, on the way up
    and on the way down. Anything that falls in a gap or outside the data is left empty with a
    note, rather than guessed.
    """
    days = days_since(grid, grid[0])
    valid = ~np.isnan(smoothed)
    run_of = np.full(len(smoothed), -1)
    for k, run in enumerate(runs(valid)):
        run_of[run] = k
    reach = round(180 / args.step)  # look this far either side of a peak for its bases

    rows = []
    for season, w0, w1 in season_windows(grid[0], grid[-1]):
        row = {"field": field, "season": season}
        inside = np.flatnonzero(valid & (grid >= w0) & (grid <= w1))
        if len(inside) == 0:
            rows.append({**row, "note": "no data"})
            continue
        ip = inside[np.argmax(smoothed[inside])]
        peak = smoothed[ip]
        if any(0 <= j < len(smoothed) and valid[j] and smoothed[j] > peak for j in (ip - 1, ip + 1)):
            rows.append({**row, "note": "no peak inside this season"})
            continue

        left = [j for j in range(max(ip - reach, 0), ip) if valid[j]]
        right = [j for j in range(ip + 1, min(ip + reach, len(smoothed) - 1) + 1) if valid[j]]
        il = min(left, key=lambda j: smoothed[j]) if left else None
        ir = min(right, key=lambda j: smoothed[j]) if right else None
        rise = max([peak - smoothed[j] for j in (il, ir) if j is not None], default=0)
        if rise < args.min_amplitude:
            rows.append({**row, "note": f"NDVI rises less than {args.min_amplitude}: no clear crop"})
            continue

        notes = []
        if ip - 1 < 0 or not valid[ip - 1] or ip + 1 >= len(smoothed) or not valid[ip + 1]:
            notes.append("peak next to a gap or the data edge")
        run = runs(valid)[run_of[ip]]
        start = end = np.nan
        # Measure a side only if the curve really climbs from (or falls to) its base there;
        # a small wiggle next to the peak is not the start or end of a season.
        if (il is None or run_of[il] != run_of[ip] or il == run[0]
                or peak - smoothed[il] < args.min_amplitude):
            notes.append("start not observed")
        else:
            start = crossing(days, smoothed, ip, il, smoothed[il] + 0.5 * (peak - smoothed[il]))
        if (ir is None or run_of[ir] != run_of[ip] or ir == run[-1]
                or peak - smoothed[ir] < args.min_amplitude):
            notes.append("end not observed")
        else:
            end = crossing(days, smoothed, ip, ir, smoothed[ir] + 0.5 * (peak - smoothed[ir]))

        as_date = lambda d: (grid[0] + timedelta(days=float(d))).date() if not np.isnan(d) else None
        in_season = (days >= start) & (days <= end)
        rows.append({**row, "_ip": ip, "peak_date": grid[ip].date(), "peak": round(float(peak), 3),
                     "start": as_date(start), "end": as_date(end),
                     "length_days": round(end - start) if not np.isnan(start + end) else None,
                     "ndvi_days": round(float(np.trapezoid(smoothed[in_season], days[in_season])), 1)
                     if not np.isnan(start + end) else None,
                     "note": "; ".join(notes)})

    # Two neighbouring peaks are separate crops only if the curve drops clearly between them.
    # Otherwise it's one crop whose peak happens to straddle a season boundary: keep the higher one.
    kept = None
    for r in [r for r in rows if "_ip" in r]:
        if kept is not None:
            between = smoothed[kept["_ip"]:r["_ip"] + 1]
            if min(kept["peak"], r["peak"]) - np.nanmin(between) < args.min_amplitude:
                lower, higher = (kept, r) if kept["peak"] < r["peak"] else (r, kept)
                for key in ("peak_date", "peak", "start", "end", "length_days", "ndvi_days"):
                    lower.pop(key, None)
                lower["note"] = f"same crop as {higher['season']} (no clear drop between the peaks)"
                kept = higher
                continue
        kept = r
    for r in rows:
        r.pop("_ip", None)
    return rows


def plot(results, obs_all, grid, pheno, args, window, mae, path):
    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.2 * len(names) + 1), sharex=True, squeeze=False)
    gdates = grid.to_pydatetime()
    for ax, color, name in zip(axes[:, 0], COLORS, names):
        gridded, smoothed = results[name]["NDVI"]
        shade_seasons(ax, grid[0].date(), grid[-1].date())
        mine = obs_all[obs_all.field == name]
        ok, flagged = mine[mine.flag == "ok"], mine[mine.flag != "ok"]

        # grey bands where the grid has no value inside the observed period: gaps too long to fill
        inside = (grid >= mine.date.min()) & (grid <= mine.date.max())
        for run in runs(np.isnan(gridded) & inside):
            ax.axvspan(gdates[max(run[0] - 1, 0)], gdates[min(run[-1] + 1, len(grid) - 1)],
                       color="grey", alpha=0.18, lw=0)

        ax.scatter(ok.date, ok.NDVI_median, s=12, color=color, alpha=0.55, lw=0, zorder=3)
        ax.scatter(flagged.date, flagged.NDVI_median, s=22, facecolor="white", edgecolor="grey", lw=1, zorder=3)
        if window:
            ax.plot(gdates, gridded, color=color, lw=0.9, ls=":", alpha=0.9)
        ax.plot(gdates, smoothed, color=color, lw=2.2)

        for p in pheno:
            if p["field"] != name or not p.get("peak_date"):
                continue
            ax.plot(pd.Timestamp(p["peak_date"]), p["peak"], marker="v", color="black", ms=6, zorder=4)
            ax.annotate(p["season"].replace("_", " "), (pd.Timestamp(p["peak_date"]), p["peak"]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
            for key, mark in (("start", ">"), ("end", "<")):
                if p.get(key):
                    t = pd.Timestamp(p[key])
                    ax.plot(t, value_at(days_since(grid, grid[0]), smoothed, days_since([t], grid[0])[0]),
                            marker=mark, color="black", ms=6, zorder=4)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("NDVI")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

    ax = axes[-1, 0]
    ax.set_xlim(grid[0] - timedelta(days=5), grid[-1] + timedelta(days=5))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    label = f"Savitzky–Golay window {window} ({window * args.step} days)" if window else "no smoothing"
    height = fig.get_figheight()
    fig.suptitle(f"{args.step}-day NDVI series  ·  {label}  ·  hold-out error {mae[window]:.3f} NDVI",
                 fontsize=12, y=1 - 0.12 / height, va="top")
    handles = [Line2D([], [], color="grey", lw=2.2, label="smoothed series" if window else "interpolated series")]
    if window:
        handles.append(Line2D([], [], color="grey", lw=0.9, ls=":", label="interpolated only"))
    handles += [
        Line2D([], [], ls="", marker="o", ms=4, color="grey", alpha=0.6, label="clear observation"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="grey", label="flagged, not used"),
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.18, label=f"gap over {args.max_gap} days, left empty"),
        Line2D([], [], ls="", marker=">", color="black", label="season start / end / peak"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / height), ncol=6,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    obs_all = pd.read_csv(args.csv, parse_dates=["date"])
    if "flag" not in obs_all.columns:
        raise SystemExit(f"{args.csv} has no flag column; run s2_fields.py first")
    obs = obs_all[obs_all.flag == "ok"]
    names = list(dict.fromkeys(obs_all.field))
    indices = [i for i in INDICES if f"{i}_median" in obs.columns]

    grid = make_grid(obs, args)
    grid_days = days_since(grid, grid[0])
    per_field = {f: obs[obs.field == f].sort_values("date") for f in names}
    print(f"{len(obs)} clear observations ({len(obs_all) - len(obs)} flagged ones dropped) for {len(names)} fields")
    print(f"Grid: {len(grid)} dates every {args.step} days, {grid[0]:%d %b %Y} to {grid[-1]:%d %b %Y}")

    # Hold-out validation: which smoothing window rebuilds hidden observations best?
    ndvi_obs = {f: (days_since(d.date, grid[0]), d.NDVI_median.to_numpy()) for f, d in per_field.items()}
    candidates = [None] + sorted(set(WINDOWS) | ({args.window} if args.window else set()))
    mae, errors = holdout(ndvi_obs, grid_days, args, candidates)
    print("\nHold-out validation on NDVI (10% of clear observations hidden, 20 repeats):")
    for w in candidates:
        label = "interpolation only" if w is None else f"window {w:2d} ({w * args.step:2d} days)"
        print(f"  {label:22} mean absolute error {mae[w]:.4f}")
    window = args.window if args.window else min(mae, key=mae.get)
    print(f"  -> using {'no smoothing' if window is None else f'window {window}'}"
          + (" (set with --window)" if args.window else " (lowest error)"))
    unique = {(f, d): e for f, d, e in errors[window]}  # the same point is often hidden in several repeats
    worst = sorted(((f, d, e) for (f, d), e in unique.items()), key=lambda x: -x[2])[:5]
    print("  largest errors: " + ", ".join(
        f"{f.split('_')[-1]} {(grid[0] + timedelta(days=d)):%d %b} {e:.2f}" for f, d, e in worst))

    # Build every field's series for every index.
    results, long_rows = {}, []
    for name, d in per_field.items():
        results[name] = {}
        obs_days = days_since(d.date, grid[0])
        for idx in indices:
            gridded = regularise(obs_days, d[f"{idx}_median"].to_numpy(), grid_days, args.max_gap)
            smoothed = smooth(gridded, window, args.order)
            results[name][idx] = (gridded, smoothed)
            long_rows += [{"field": name, "date": t.date(), "index": idx,
                           "interpolated": None if np.isnan(g) else round(g, 4),
                           "smoothed": None if np.isnan(s) else round(s, 4)}
                          for t, g, s in zip(grid, gridded, smoothed)]
        gaps = np.diff(obs_days)
        long_gaps = [(d.date.iloc[i], d.date.iloc[i + 1]) for i in np.flatnonzero(gaps > args.max_gap)]
        if long_gaps:
            print(f"  {name}: left empty " + ", ".join(f"{a:%d %b} to {b:%d %b} ({(b - a).days} days)" for a, b in long_gaps))

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(long_rows).to_csv(os.path.join(args.out, "series.csv"), index=False)
    matrix = pd.DataFrame({f"{idx}_{t:%Y-%m-%d}": [results[n][idx][1][k] for n in names]
                           for idx in indices for k, t in enumerate(grid)}, index=pd.Index(names, name="field"))
    matrix.round(4).to_csv(os.path.join(args.out, "series_matrix.csv"))

    pheno = [row for name in names for row in phenology(name, grid, results[name]["NDVI"][1], args)]
    pd.DataFrame(pheno).to_csv(os.path.join(args.out, "phenology.csv"), index=False)
    print("\nCrop seasons from the NDVI series (start and end = halfway between base and peak):")
    for p in pheno:
        if p.get("peak_date"):
            span = (f"{p['start']:%d %b} -> {p['end']:%d %b}, {p['length_days']} days" if p.get("length_days")
                    else f"{p['start']:%d %b} -> ?" if p.get("start") else f"? -> {p['end']:%d %b}" if p.get("end") else "? -> ?")
            print(f"  {p['field']:18} {p['season']:12} peak {p['peak']:.2f} on {p['peak_date']:%d %b %Y}   {span}"
                  + (f"   ({p['note']})" if p["note"] else ""))
        else:
            print(f"  {p['field']:18} {p['season']:12} {p['note']}")

    if len(names) <= 12:
        plot(results, obs_all, grid, pheno, args, window, mae, os.path.join(args.out, "series.png"))
    else:
        print(f"\n{len(names)} fields: series.png skipped (one panel per field would be unreadable)")
    print(f"\nMatrix: {len(names)} fields x {matrix.shape[1]} columns ({len(indices)} indices x {len(grid)} dates)")
    print(f"Saved series.csv, series_matrix.csv, phenology.csv and series.png in ./{args.out}/")


if __name__ == "__main__":
    main()
