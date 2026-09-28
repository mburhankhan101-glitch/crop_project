"""
Spectral indices per field, from free Sentinel-2 images on Microsoft Planetary Computer.

Reads field boundaries from a GeoJSON file, shrinks each field inwards so only pixels
entirely inside it count, and tracks every index in s2_indices.INDICES field by field.

The SCL cloud mask misses cloud edges and haze, so each observation is also checked and
flagged (not dropped):
    cloud edges  SCL cloud and shadow pixels are grown by --cloud-buffer pixels first
    haze         median blue minus red above --haze; haze brightens blue more than red
    dip          NDVI more than --dip below the observations before and after it, both within
                 --dip-days; crops don't lose that much greenness and regrow within days

    python s2_fields.py                                        # fields.geojson, all indices
    python s2_fields.py --fields my_fields.geojson --indices NDVI,NDMI
    python s2_fields.py --buffer 20                            # shrink fields by 20 m instead of 10 m
    python s2_fields.py --cloud-buffer 0 --haze 1              # plain SCL mask, no haze flags

Downloaded pixel windows are cached in ./cache/, so a second run only downloads new scenes
and takes seconds. Use --no-cache to bypass it.

Writes to ./output/:
    fields.csv             one row per field per clear date: index statistics, haze score and flag
    fields_timeseries.png  one panel per index, one line per field (median); flagged points hollow
    fields_flagged.png     true-colour thumbnails of every flagged date, to check each one by eye
    fields_map.png         field boundaries on true colour and NDVI for the latest date all fields were clear
"""
import argparse
import csv
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

from s2_indices import CLEAR_SCL, INDICES, PIXEL_M, bands_for, reflectance, search_geometry, shade_seasons

# Reading windows are snapped to the 20 m grid, so every 20 m pixel covers exactly 2 x 2
# pixels of the 10 m grid. Sentinel-2 tile corners are multiples of 20 m, so this is exact.
GRID_M = 20
TWENTY_M_BANDS = {"B05", "B06", "B07", "B8A", "B11", "B12"}
COLORS = plt.cm.tab10.colors

# Pixel windows already downloaded are kept here; None switches caching off (--no-cache).
CACHE_DIR = "cache"

# SCL classes that get grown outwards: cloud shadow, medium and high probability cloud, thin cirrus.
CLOUD_SCL = [3, 8, 9, 10]
HAZE_BANDS = {"B02", "B04"}


def parse_args():
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default="fields.geojson", help="GeoJSON with one polygon per field")
    p.add_argument("--name", default="name", help="property holding each field's name (default: name)")
    p.add_argument("--buffer", type=float, default=10, help="shrink each field inwards by this many metres (default 10)")
    p.add_argument("--indices", default=",".join(INDICES), help=f"comma-separated, from {', '.join(INDICES)}")
    p.add_argument("--start", default=str(today - timedelta(days=395)), help="YYYY-MM-DD (default: ~13 months ago)")
    p.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    p.add_argument("--max-cloud", type=float, default=50, help="skip scenes with more cloud than this %% overall")
    p.add_argument("--min-clear", type=float, default=0.8, help="fraction of a field that must be cloud-free")
    p.add_argument("--cloud-buffer", type=int, default=2,
                   help="grow SCL cloud and shadow by this many 10 m pixels (default 2)")
    p.add_argument("--haze", type=float, default=0.008,
                   help="flag haze when a field's median blue minus red exceeds this (default 0.008, "
                        "calibrated on the fields near Raiwind)")
    p.add_argument("--dip", type=float, default=0.1, help="flag an NDVI dip this far below both neighbours")
    p.add_argument("--dip-days", type=int, default=20, help="neighbours must be within this many days")
    p.add_argument("--out", default="output", help="output folder")
    p.add_argument("--cache", default="cache", help="folder for downloaded pixel windows (default: cache)")
    p.add_argument("--no-cache", action="store_true", help="always download, and don't save anything")
    args = p.parse_args()

    args.indices = [name.strip().upper() for name in args.indices.split(",") if name.strip()]
    unknown = [name for name in args.indices if name not in INDICES]
    if unknown:
        p.error(f"unknown index {', '.join(unknown)}; choose from {', '.join(INDICES)}")
    if "NDVI" not in args.indices:
        args.indices.insert(0, "NDVI")  # the dip check needs it
    return args


def load_fields(args):
    """Polygons from the GeoJSON, validated, with unique names."""
    fields = gpd.read_file(args.fields)
    if args.name not in fields.columns:
        raise SystemExit(f"{args.fields} has no '{args.name}' property; use --name to pick another")

    polygons = fields.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not polygons.all():
        print(f"Ignoring {int((~polygons).sum())} feature(s) that aren't polygons (points or lines).")
    fields = fields[polygons].reset_index(drop=True)
    if fields.empty:
        raise SystemExit(f"No polygons in {args.fields}")

    duplicates = fields[args.name][fields[args.name].duplicated()].tolist()
    if duplicates:
        raise SystemExit(f"Field names must be unique; repeated: {', '.join(map(str, duplicates))}")
    fields["geometry"] = fields.geometry.make_valid()  # repairs self-crossing boundaries
    return fields


def item_crs(item):
    return item.properties.get("proj:code") or f"EPSG:{item.properties['proj:epsg']}"


def snapped_grid(bounds, res):
    """Grid of res-metre pixels covering bounds, snapped outwards to the 20 m grid: (transform, shape)."""
    x0, y0, x1, y1 = bounds
    x0, y0 = math.floor(x0 / GRID_M) * GRID_M, math.floor(y0 / GRID_M) * GRID_M
    x1, y1 = math.ceil(x1 / GRID_M) * GRID_M, math.ceil(y1 / GRID_M) * GRID_M
    return Affine(res, 0, x0, 0, -res, y1), (round((y1 - y0) / res), round((x1 - x0) / res))


def pure_pixels(geom, res):
    """How many pixel centres of a res-metre grid fall inside the geometry."""
    if geom.is_empty:
        return 0
    transform, shape = snapped_grid(geom.bounds, res)
    return int(geometry_mask([geom], out_shape=shape, transform=transform, invert=True).sum())


def cache_path(item, band, x0, y0, x1, y1):
    """Where one band's window of one scene is cached.

    The key holds everything that decides the content: the scene (whose ID includes ESA's
    processing time, so a reprocessed scene gets a new ID and never hits a stale file),
    the band, and the window. Everything computed from the pixels (indices, masks, flags)
    stays out of the key, so changing that code never invalidates the cache.
    """
    return os.path.join(CACHE_DIR, item.id, f"{band}_{x0:.0f}_{y0:.0f}_{x1:.0f}_{y1:.0f}.npy")


def read_bounds(item, bounds, bands):
    """Read every band over bounds (in the tile's coordinates) onto one 10 m grid.

    20 m bands are upsampled with nearest neighbour. Returns (arrays, transform),
    or None if the area runs off the edge of this tile. Windows are cached on disk
    unless CACHE_DIR is None.
    """
    transform, shape = snapped_grid(bounds, PIXEL_M)
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + shape[1] * PIXEL_M, y1 - shape[0] * PIXEL_M

    out = {}
    for band in bands:
        path = cache_path(item, band, x0, y0, x1, y1) if CACHE_DIR else None
        if path and os.path.exists(path):
            out[band] = np.load(path)
            continue
        with rasterio.open(item.assets[band].href) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            inside = (win.col_off >= 0 and win.row_off >= 0
                      and win.col_off + win.width <= src.width and win.row_off + win.height <= src.height)
            if not inside:
                return None
            out[band] = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
        if path:
            # Write to a temporary name first so a crash or a parallel thread never leaves half a file.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, out[band])
            os.replace(tmp, path)
    return out, transform


def field_masks(fields, crs, args):
    """Where the (shrunk) fields sit on the reading grid, in one tile coordinate system.

    Every scene in that coordinate system is read over the same window, so the masks are
    computed once, here, rather than per scene. (rasterio's rasterize isn't safe to call
    from several threads at once; it can occasionally return an empty mask.)
    """
    inner = fields.to_crs(crs).geometry.buffer(-args.buffer)
    # Read a margin around the fields so clouds just outside them can be grown inwards.
    margin = args.cloud_buffer * PIXEL_M
    bounds = inner[~inner.is_empty].total_bounds + np.array([-margin, -margin, margin, margin])
    transform, shape = snapped_grid(bounds, PIXEL_M)
    masks = [None if g.is_empty else geometry_mask([g], out_shape=shape, transform=transform, invert=True)
             for g in inner]
    return bounds, masks


def grow(mask, pixels):
    """Grow a boolean mask outwards by `pixels` in every direction, diagonals included."""
    if pixels <= 0:
        return mask
    size = 2 * pixels + 1
    return sliding_window_view(np.pad(mask, pixels), (size, size)).any(axis=(-2, -1))


def scene_fields(item, names, layouts, args):
    """One row per field that is clear enough on this date."""
    bands = sorted(set(bands_for(args.indices)) | HAZE_BANDS)
    bounds, masks = layouts[item_crs(item)]
    try:
        read = read_bounds(item, bounds, bands + ["SCL"])
    except rasterio.errors.RasterioIOError as e:
        print(f"  {item.datetime.date()}: could not read ({e})")
        return []
    if read is None:
        return []
    arrays, _ = read

    # SCL often misses the thin fringe around clouds and shadows, so grow them first.
    cloud = grow(np.isin(arrays["SCL"], CLOUD_SCL), args.cloud_buffer)
    valid = np.isin(arrays["SCL"], CLEAR_SCL) & ~cloud
    for band in bands:
        valid &= arrays[band] > 0
    refl = {band: reflectance(arrays[band], item) for band in bands}
    values = {name: INDICES[name]["formula"](refl) for name in args.indices}
    blue_minus_red = refl["B02"] - refl["B04"]

    rows = []
    for name, inside in zip(names, masks):
        if inside is None:
            continue
        use = inside & valid
        clear = use.sum() / max(inside.sum(), 1)
        if not use.any() or clear < args.min_clear:
            continue
        row = {"field": name, "date": item.datetime.date(), "pixels": int(use.sum()),
               "clear_fraction": float(clear), "haze_score": float(np.median(blue_minus_red[use])),
               "item": item}
        for idx in args.indices:
            v = values[idx][use]
            row[f"{idx}_mean"] = float(v.mean())
            row[f"{idx}_median"] = float(np.median(v))
            row[f"{idx}_std"] = float(v.std())
        rows.append(row)
    return rows


def add_flags(rows, names, args):
    """Mark suspicious observations with the reasons, instead of dropping them.

    rows must be sorted by date. Sets row["flag"] to "ok", or reasons joined with "+":
    haze, dip, scene_haze.
    """
    for name in names:
        mine = [r for r in rows if r["field"] == name]
        for i, r in enumerate(mine):
            reasons = []
            if r["haze_score"] > args.haze:
                reasons.append("haze")
            if 0 < i < len(mine) - 1:
                before, after = mine[i - 1], mine[i + 1]
                close = ((r["date"] - before["date"]).days <= args.dip_days
                         and (after["date"] - r["date"]).days <= args.dip_days)
                v = r["NDVI_median"]
                if close and v < before["NDVI_median"] - args.dip and v < after["NDVI_median"] - args.dip:
                    reasons.append("dip")
            r["flag"] = "+".join(reasons) or "ok"

    # Haze is in the air, not in a field. The blue - red test only works over dense crops
    # (soil pulls it far below zero), so one hazy field marks every field on that date.
    # This assumes the fields lie within a few km of each other.
    hazy_dates = {r["date"] for r in rows if "haze" in r["flag"].split("+")}
    for r in rows:
        reasons = r["flag"].split("+")
        if r["date"] in hazy_dates and "haze" not in reasons:
            r["flag"] = "+".join(["scene_haze"] + [x for x in reasons if x != "ok"])


def plot_timeseries(rows, names, args, path):
    dates = sorted({r["date"] for r in rows})
    n = len(args.indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n + 0.8), sharex=True, squeeze=False)

    for ax, idx in zip(axes[:, 0], args.indices):
        spec = INDICES[idx]
        shade_seasons(ax, dates[0], dates[-1])
        for color, name in zip(COLORS, names):
            ok = [r for r in rows if r["field"] == name and r["flag"] == "ok"]
            flagged = [r for r in rows if r["field"] == name and r["flag"] != "ok"]
            # The line runs through trusted points only; flagged ones stay visible but hollow.
            ax.plot([r["date"] for r in ok], [r[f"{idx}_median"] for r in ok],
                    "-o", color=color, ms=3, lw=1.5, label=name)
            ax.scatter([r["date"] for r in flagged], [r[f"{idx}_median"] for r in flagged],
                       s=30, facecolor="white", edgecolor=color, lw=1.4, zorder=3)
        for y, text in spec["lines"]:
            ax.axhline(y, color="grey", lw=0.8, ls=":")
            ax.text(dates[0], y, f" {text}", fontsize=7.5, color="grey", va="bottom")
        ax.set_ylim(*spec["ylim"])
        ax.set_ylabel(idx, fontsize=11, fontweight="bold")
        ax.set_title(spec["about"], fontsize=8.5, color="#555", loc="left")
        ax.grid(alpha=0.25)

    ax = axes[-1, 0]
    ax.set_xlim(dates[0] - timedelta(days=7), dates[-1] + timedelta(days=7))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    height = fig.get_figheight()
    n_flagged = sum(r["flag"] != "ok" for r in rows)
    fig.suptitle(f"Median of each index per field  ·  fields shrunk by {args.buffer:g} m  ·  "
                 f"{len(dates)} dates  ·  {n_flagged} observations flagged",
                 fontsize=12, y=1 - 0.12 / height, va="top")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="grey", mew=1.4,
                       label="flagged (haze, scene haze or dip), not joined"),
                mpatches.Patch(color="#f2c14e", alpha=0.35, label="Rabi (Nov–Apr)"),
                mpatches.Patch(color="#4e9af2", alpha=0.3, label="Kharif (May–Oct)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / height),
               fontsize=8, ncol=min(len(handles), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_outline(ax, geom, transform, **style):
    """Draw a polygon's outline on an image axis whose pixels follow transform."""
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        cols, rows = ~transform * (np.array(xs), np.array(ys))
        ax.plot(cols - 0.5, rows - 0.5, **style)


def true_colour(item, bounds):
    """Brightened true-colour image over bounds, and its transform; None off the tile edge."""
    read = read_bounds(item, bounds, ["B02", "B03", "B04"])
    if read is None:
        return None
    arrays, transform = read
    refl = {band: reflectance(arrays[band], item) for band in arrays}
    return np.clip(np.dstack([refl["B04"], refl["B03"], refl["B02"]]) * 3.5, 0, 1), transform


def plot_flagged(rows, fields, args, path):
    """True-colour thumbnails of every flagged date, next to the clearest date for comparison.

    All thumbnails share one brightness stretch, so haze shows up as a milky, washed-out look.
    """
    flagged = sorted({r["date"] for r in rows if r["flag"] != "ok"})
    if not flagged:
        return False
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    all_ok = [d for d, rs in by_date.items() if len(rs) == len(fields) and all(r["flag"] == "ok" for r in rs)]
    reference = min(all_ok, key=lambda d: max(r["haze_score"] for r in by_date[d])) if all_ok else None
    dates = ([reference] if reference else []) + flagged

    def thumbnail(d):
        item = by_date[d][0]["item"]
        outer = fields.to_crs(item_crs(item)).geometry
        return true_colour(item, outer.buffer(150).total_bounds), outer

    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(thumbnail, dates))

    ncols = min(len(dates), 4)
    nrows = math.ceil(len(dates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.1 * nrows), squeeze=False)
    for ax, d, (image, outer) in zip(axes.ravel(), dates, images):
        ax.set_xticks([])
        ax.set_yticks([])
        if image is None:
            ax.set_title(f"{d}\n(off the tile edge)", fontsize=8)
            continue
        rgb, transform = image
        ax.imshow(rgb)
        for color, geom in zip(COLORS, outer):
            draw_outline(ax, geom, transform, color=color, lw=1.4)
        if d == reference:
            ax.set_title(f"{d}\nclearest date, for comparison", fontsize=8, fontweight="bold")
        else:
            notes = [f"{r['field'].split('_')[-1]}: {r['flag']} (NDVI {r['NDVI_median']:.2f})"
                     for r in by_date[d] if r["flag"] != "ok"]
            ax.set_title(f"{d}\n" + "\n".join(notes), fontsize=7.5)
    for ax in axes.ravel()[len(dates):]:
        ax.axis("off")
    fig.suptitle("Flagged observations in true colour, all at the same brightness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def plot_map(item, fields, args, path):
    """True colour and NDVI around the fields, with full boundaries and the shrunk area used."""
    crs = item_crs(item)
    outer = fields.to_crs(crs).geometry
    inner = outer.buffer(-args.buffer)
    read = read_bounds(item, outer.buffer(250).total_bounds, ["B02", "B03", "B04", "B08"])
    if read is None:
        print("  map area runs off the tile edge; skipping fields_map.png")
        return
    arrays, transform = read
    refl = {band: reflectance(arrays[band], item) for band in arrays}
    rgb = np.clip(np.dstack([refl["B04"], refl["B03"], refl["B02"]]) * 3.5, 0, 1)
    ndvi = INDICES["NDVI"]["formula"](refl)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(rgb)
    axes[0].set_title("True colour")
    im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    axes[1].set_title("NDVI")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        for color, name, full, shrunk in zip(COLORS, fields[args.name], outer, inner):
            draw_outline(ax, full, transform, color="white", lw=2.4)
            draw_outline(ax, full, transform, color=color, lw=1.3)
            draw_outline(ax, shrunk, transform, color="white", lw=0.9, ls=":")
            cx, cy = ~transform * (full.centroid.x, full.centroid.y)
            ax.text(cx, cy, name, ha="center", va="center", fontsize=7.5,
                    bbox=dict(fc="white", ec=color, alpha=0.85, lw=1))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sentinel-2, {item.datetime.date()}  ·  solid = field boundary  ·  "
                 f"dotted = shrunk by {args.buffer:g} m (pixels actually used)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    global CACHE_DIR
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    CACHE_DIR = None if args.no_cache else args.cache
    started = time.monotonic()
    fields = load_fields(args)
    names = list(fields[args.name])

    # How many pixels each field really has, before downloading anything.
    utm = fields.to_crs(fields.estimate_utm_crs())
    twenty_m = [i for i in args.indices if set(INDICES[i]["bands"]) & TWENTY_M_BANDS]
    print(f"{len(fields)} fields from {args.fields}, shrunk by {args.buffer:g} m:")
    print(f"  {'field':22} {'acres':>6} {'pure 10 m px':>13} {'pure 20 m px':>13}")
    for name, geom in zip(names, utm.geometry):
        shrunk = geom.buffer(-args.buffer)
        n10, n20 = pure_pixels(shrunk, 10), pure_pixels(shrunk, 20)
        note = "  <- too small, skipped" if n10 == 0 else ("  <- few 20 m pixels" if twenty_m and n20 < 10 else "")
        print(f"  {name:22} {geom.area / 4047:6.1f} {n10:13d} {n20:13d}{note}")
    if twenty_m:
        print(f"  ({', '.join(twenty_m)} use 20 m bands, so they rest on the 20 m count.)")

    area = mapping(box(*fields.to_crs("EPSG:4326").total_bounds))
    print(f"\nSearching Sentinel-2 scenes from {args.start} to {args.end} ...")
    items = search_geometry(area, args.start, args.end, args.max_cloud)
    layouts = {crs: field_masks(fields, crs, args) for crs in {item_crs(it) for it in items}}
    print(f"Found {len(items)} passes with under {args.max_cloud:g}% cloud. "
          f"Reading {', '.join(bands_for(args.indices))} + SCL ...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for batch in pool.map(lambda it: scene_fields(it, names, layouts, args), items) for r in batch]
    rows.sort(key=lambda r: (r["date"], names.index(r["field"])))
    if not rows:
        print("No clear observations. Try a longer date range, a higher --max-cloud or a smaller --buffer.")
        return
    add_flags(rows, names, args)

    csv_path = os.path.join(args.out, "fields.csv")
    stats = [f"{idx}_{stat}" for idx in args.indices for stat in ("mean", "median", "std")]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "date", "pixels", "clear_fraction", "haze_score", "flag"] + stats)
        for r in rows:
            w.writerow([r["field"], r["date"], r["pixels"], f"{r['clear_fraction']:.2f}",
                        f"{r['haze_score']:.4f}", r["flag"]] + [f"{r[s]:.4f}" for s in stats])

    plot_timeseries(rows, names, args, os.path.join(args.out, "fields_timeseries.png"))
    plot_flagged(rows, fields, args, os.path.join(args.out, "fields_flagged.png"))
    all_clear = [d for d in sorted({r["date"] for r in rows}, reverse=True)
                 if sum(r["date"] == d and r["flag"] == "ok" for r in rows) == len(names)]
    if all_clear:
        item = next(r["item"] for r in rows if r["date"] == all_clear[0])
        plot_map(item, fields, args, os.path.join(args.out, "fields_map.png"))

    print("\nFlagged observations (kept in fields.csv, drawn hollow on the chart):")
    for r in rows:
        if r["flag"] != "ok":
            print(f"  {r['field']:22} {r['date']}  {r['flag']:9} NDVI {r['NDVI_median']:.2f}  "
                  f"haze score {r['haze_score']:+.3f}")

    idx = args.indices[0]
    print(f"\n{'field':22} {'dates':>5}   {idx} median: {'peak':>17} {'lowest':>17} {'latest':>17}")
    for name in names:
        mine = [r for r in rows if r["field"] == name]
        if not mine:
            print(f"{name:22} {0:5d}   no clear dates")
            continue
        key = f"{idx}_median"
        cell = lambda r: f"{r[key]:+.2f} {r['date']}"
        hi, lo = max(mine, key=lambda r: r[key]), min(mine, key=lambda r: r[key])
        print(f"{name:22} {len(mine):5d}   {'':{len(idx) + 8}} {cell(hi):>17} {cell(lo):>17} {cell(mine[-1]):>17}")
    print(f"\nSaved fields.csv, fields_timeseries.png, fields_flagged.png and fields_map.png in ./{args.out}/ "
          f"in {time.monotonic() - started:.0f} s" + (f" (pixel cache: ./{CACHE_DIR}/)" if CACHE_DIR else ""))


if __name__ == "__main__":
    main()
