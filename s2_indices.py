"""
Spectral indices over time for a patch of farmland, from free Sentinel-2
satellite images hosted on Microsoft Planetary Computer. No account needed.

    NDVI  greenness                 NIR vs red                    10 m
    NDWI  open water                green vs NIR                  10 m
    NDMI  moisture in the canopy    NIR vs short-wave infrared    20 m
    EVI   greenness, less saturated NIR, red and blue             10 m

Default spot: farmland between Raiwind and Kasur, south-west of Lahore.

    python s2_indices.py                            # default Lahore farmland, all indices
    python s2_indices.py --lat 31.30 --lon 74.07    # any other spot
    python s2_indices.py --indices NDVI,NDMI        # only some indices
    python s2_indices.py --start 2024-01-01 --end 2024-12-31

Writes to ./output/:
    indices.csv             one row per clear satellite pass
    indices_timeseries.png  one panel per index over the season
    indices_map.png         true colour plus a map of each index for the latest clear pass
"""
import argparse
import csv
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

# Remote reads of cloud-optimised GeoTIFFs: skip directory listings, retry flaky requests,
# and give up on a stalled connection instead of hanging forever.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "15")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PIXEL_M = 10  # everything is analysed on the 10 m grid of the sharpest bands

# Scene Classification Layer (SCL) values we trust: 4 vegetation, 5 bare soil, 6 water.
# The rest are cloud, cloud shadow, thin cirrus, snow or missing data.
CLEAR_SCL = [4, 5, 6]

# Planetary Computer supports the cloud-cover "query" filter but doesn't advertise it.
warnings.filterwarnings("ignore", message=".*does not conform to QUERY")


def nd(a, b):
    """Normalised difference: how much a exceeds b, scaled to -1..1 and independent of brightness."""
    return (a - b) / (a + b)


# Each index: the bands it needs, its formula on surface reflectance (0-1), and how to draw it.
# To add an index, add an entry here; the rest of the script picks it up.
INDICES = {
    "NDVI": {
        "about": "greenness: leaves absorb red and reflect near-infrared",
        "bands": ["B04", "B08"],
        "formula": lambda r: nd(r["B08"], r["B04"]),
        "ylim": (-0.1, 1.0),
        "lines": [(0.2, "bare soil / harvested below this"), (0.6, "dense healthy crop above this")],
        "cmap": ("RdYlGn", -0.1, 0.9),
    },
    "NDWI": {
        "about": "open water: water reflects some green but almost no near-infrared",
        "bands": ["B03", "B08"],
        "formula": lambda r: nd(r["B03"], r["B08"]),
        "ylim": (-0.9, 0.5),
        "lines": [(0.0, "above 0: standing water, e.g. flooded rice paddies")],
        "cmap": ("BrBG", -0.6, 0.4),
    },
    "NDMI": {
        "about": "moisture: water inside leaves and soil absorbs short-wave infrared (20 m bands)",
        "bands": ["B8A", "B11"],
        "formula": lambda r: nd(r["B8A"], r["B11"]),
        "ylim": (-0.4, 0.7),
        "lines": [(0.0, "below 0: dry soil or a drying crop")],
        "cmap": ("BrBG", -0.3, 0.6),
    },
    "EVI": {
        "about": "greenness that saturates less in dense crops and is less affected by haze",
        "bands": ["B02", "B04", "B08"],
        # The constants assume reflectance on a 0-1 scale, so raw digital numbers would give nonsense.
        "formula": lambda r: 2.5 * (r["B08"] - r["B04"]) / (r["B08"] + 6 * r["B04"] - 7.5 * r["B02"] + 1),
        "ylim": (-0.1, 1.0),
        "lines": [(0.2, "bare soil / harvested below this")],
        "cmap": ("RdYlGn", -0.1, 0.9),
    },
    "NDRE": {
        "about": "chlorophyll via the red edge: keeps responding in dense crops where NDVI flattens (20 m bands)",
        "bands": ["B8A", "B05"],
        "formula": lambda r: nd(r["B8A"], r["B05"]),
        "ylim": (-0.1, 0.8),
        "lines": [],
        "cmap": ("RdYlGn", -0.1, 0.7),
    },
}


def parse_args():
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=31.1923, help="latitude of the centre (default: farmland SW of Lahore)")
    p.add_argument("--lon", type=float, default=74.3019, help="longitude of the centre")
    p.add_argument("--indices", default=",".join(INDICES), help=f"comma-separated, from {', '.join(INDICES)}")
    p.add_argument("--size", type=int, default=400, help="side of the square to average, in metres (default 400)")
    p.add_argument("--map-size", type=int, default=2000, help="side of the square drawn in indices_map.png, in metres")
    p.add_argument("--start", default=str(today - timedelta(days=395)), help="YYYY-MM-DD (default: ~13 months ago)")
    p.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    p.add_argument("--max-cloud", type=float, default=50, help="skip scenes with more cloud than this %% overall")
    p.add_argument("--min-clear", type=float, default=0.8, help="fraction of the square that must be cloud-free")
    p.add_argument("--out", default="output", help="output folder")
    args = p.parse_args()

    args.indices = [name.strip().upper() for name in args.indices.split(",") if name.strip()]
    unknown = [name for name in args.indices if name not in INDICES]
    if unknown:
        p.error(f"unknown index {', '.join(unknown)}; choose from {', '.join(INDICES)}")
    return args


def bands_for(indices):
    """The bands the chosen indices need, each read once."""
    return sorted({band for name in indices for band in INDICES[name]["bands"]})


def search(lat, lon, start, end, max_cloud):
    """Find Sentinel-2 scenes over the point, keeping the least cloudy one per day."""
    return search_geometry({"type": "Point", "coordinates": [lon, lat]}, start, end, max_cloud)


def search_geometry(geometry, start, end, max_cloud):
    """Find Sentinel-2 scenes touching a GeoJSON geometry, keeping the least cloudy one per day."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()

    # A point near a tile edge is covered by two overlapping tiles on the same pass.
    best = {}
    for item in items:
        day = item.datetime.date()
        if day not in best or item.properties["eo:cloud_cover"] < best[day].properties["eo:cloud_cover"]:
            best[day] = item
    return [best[d] for d in sorted(best)]


def read_square(item, lat, lon, size_m, bands):
    """Read a size_m x size_m square centred on (lat, lon) for each band, on a common 10 m grid.

    Coarser bands (the 20 m red-edge, narrow NIR and SWIR bands, and the 20 m SCL) are
    upsampled with nearest neighbour: each 20 m pixel becomes 2 x 2 identical 10 m pixels,
    so bands line up pixel for pixel without inventing detail.
    Returns None if the square runs off the edge of this tile.
    """
    crs = item.properties.get("proj:code") or f"EPSG:{item.properties['proj:epsg']}"
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    x, y, half = xs[0], ys[0], size_m / 2
    shape = (round(size_m / PIXEL_M),) * 2

    out = {}
    for band in bands:
        with rasterio.open(item.assets[band].href) as src:
            win = from_bounds(x - half, y - half, x + half, y + half, src.transform)
            win = win.round_offsets().round_lengths()
            inside = (win.col_off >= 0 and win.row_off >= 0
                      and win.col_off + win.width <= src.width and win.row_off + win.height <= src.height)
            if not inside:
                return None
            out[band] = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
    return out


def reflectance(dn, item):
    """Convert raw digital numbers to surface reflectance (0-1).

    Since processing baseline 04.00 (January 2022) ESA adds 1000 to every value,
    which must be removed first or every index comes out wrong.
    """
    offset = 1000 if item.properties.get("s2:processing_baseline", "00.00") >= "04.00" else 0
    return np.clip((dn.astype("float32") - offset) / 10000, 1e-4, None)


def scene_indices(item, args):
    """Mean and spread of each index over the cloud-free pixels in the square, or None if too cloudy."""
    bands = bands_for(args.indices)
    try:
        sq = read_square(item, args.lat, args.lon, args.size, bands + ["SCL"])
    except rasterio.errors.RasterioIOError as e:
        print(f"  {item.datetime.date()}: could not read ({e})")
        return None
    if sq is None:
        return None

    clear = np.isin(sq["SCL"], CLEAR_SCL)
    for band in bands:
        clear &= sq[band] > 0
    if clear.mean() < args.min_clear:
        return None

    refl = {band: reflectance(sq[band], item) for band in bands}
    row = {
        "date": item.datetime.date(),
        "clear_fraction": float(clear.mean()),
        "scene_cloud_pct": float(item.properties["eo:cloud_cover"]),
        "item": item,
    }
    for name in args.indices:
        values = INDICES[name]["formula"](refl)[clear]
        row[f"{name}_mean"] = float(values.mean())
        row[f"{name}_std"] = float(values.std())
    return row


def shade_seasons(ax, first, last):
    # Punjab's two cropping seasons: Rabi (winter, mainly wheat) and Kharif (summer, rice/cotton/maize).
    for year in range(first.year - 1, last.year + 1):
        ax.axvspan(date(year, 11, 1), date(year + 1, 4, 30), color="#f2c14e", alpha=0.12, lw=0)
        ax.axvspan(date(year + 1, 5, 1), date(year + 1, 10, 31), color="#4e9af2", alpha=0.10, lw=0)


def plot_timeseries(rows, args, path):
    dates = [r["date"] for r in rows]
    n = len(args.indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n + 0.8), sharex=True, squeeze=False)

    for ax, name in zip(axes[:, 0], args.indices):
        spec = INDICES[name]
        mean = np.array([r[f"{name}_mean"] for r in rows])
        std = np.array([r[f"{name}_std"] for r in rows])

        shade_seasons(ax, dates[0], dates[-1])
        ax.fill_between(dates, mean - std, mean + std, color="#2e7d32", alpha=0.15, lw=0)
        ax.plot(dates, mean, "-o", color="#2e7d32", ms=3.5, lw=1.6)
        for y, text in spec["lines"]:
            ax.axhline(y, color="grey", lw=0.8, ls=":")
            ax.text(dates[0], y, f" {text}", fontsize=7.5, color="grey", va="bottom")

        ax.set_ylim(*spec["ylim"])
        ax.set_ylabel(name, fontsize=11, fontweight="bold")
        ax.set_title(spec["about"], fontsize=8.5, color="#555", loc="left")
        ax.grid(alpha=0.25)

    ax = axes[-1, 0]
    ax.set_xlim(dates[0] - timedelta(days=7), dates[-1] + timedelta(days=7))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # Title and legend sit in a fixed-height header, whatever the number of panels.
    height = fig.get_figheight()
    fig.suptitle(
        f"Spectral indices at {args.lat:.4f}N, {args.lon:.4f}E  ·  {args.size} m square  ·  "
        f"{len(rows)} clear Sentinel-2 passes", fontsize=12, y=1 - 0.12 / height, va="top",
    )
    fig.legend(handles=[
        mpatches.Patch(color="#2e7d32", alpha=0.3, label="mean ± spread across the square"),
        mpatches.Patch(color="#f2c14e", alpha=0.35, label="Rabi season (Nov–Apr, wheat)"),
        mpatches.Patch(color="#4e9af2", alpha=0.3, label="Kharif season (May–Oct, rice etc.)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / height), fontsize=8, ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(item, args, path):
    """True-colour image plus one map per index, with the averaged square outlined."""
    bands = sorted(set(bands_for(args.indices)) | {"B02", "B03", "B04"})
    sq = read_square(item, args.lat, args.lon, args.map_size, bands + ["SCL"])
    if sq is None:
        print("  map area runs off the tile edge; skipping indices_map.png")
        return
    refl = {band: reflectance(sq[band], item) for band in bands}
    clear = np.isin(sq["SCL"], CLEAR_SCL)

    panels = 1 + len(args.indices)
    ncols = min(panels, 3)
    nrows = math.ceil(panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False)
    axes = axes.ravel()

    # Brighten: raw surface reflectance looks very dark on screen.
    rgb = np.clip(np.dstack([refl["B04"], refl["B03"], refl["B02"]]) * 3.5, 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title("True colour")
    for ax, name in zip(axes[1:], args.indices):
        cmap_name, vmin, vmax = INDICES[name]["cmap"]
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#b0b0b0")  # masked (cloudy) pixels are NaN; show them grey
        im = ax.imshow(np.where(clear, INDICES[name]["formula"](refl), np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    box = args.size / PIXEL_M
    corner = (rgb.shape[0] - box) / 2
    for ax in axes[:panels]:
        ax.add_patch(mpatches.Rectangle((corner, corner), box, box, fill=False, ec="white", lw=1.3, ls="--"))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[panels:]:
        ax.axis("off")

    fig.suptitle(f"Sentinel-2, {item.datetime.date()}  ·  {args.map_size / 1000:g} km across  ·  "
                 f"dashed box = averaged square  ·  grey = masked (cloud)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Searching Sentinel-2 scenes over {args.lat}, {args.lon} from {args.start} to {args.end} ...")
    items = search(args.lat, args.lon, args.start, args.end, args.max_cloud)
    print(f"Found {len(items)} passes with under {args.max_cloud:g}% cloud. "
          f"Reading {', '.join(bands_for(args.indices))} + SCL ...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(lambda it: scene_indices(it, args), items) if r is not None]
    rows.sort(key=lambda r: r["date"])
    print(f"{len(rows)} passes were clear over your square.")
    if not rows:
        print("Nothing to plot. Try a longer date range or a higher --max-cloud.")
        return

    csv_path = os.path.join(args.out, "indices.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [f"{name}_{stat}" for name in args.indices for stat in ("mean", "std")]
                   + ["clear_fraction", "scene_cloud_pct"])
        for r in rows:
            w.writerow([r["date"]] + [f"{r[f'{name}_{stat}']:.4f}" for name in args.indices for stat in ("mean", "std")]
                       + [f"{r['clear_fraction']:.2f}", f"{r['scene_cloud_pct']:.1f}"])

    plot_timeseries(rows, args, os.path.join(args.out, "indices_timeseries.png"))
    plot_map(rows[-1]["item"], args, os.path.join(args.out, "indices_map.png"))

    print(f"\n{'index':6} {'peak':>18} {'lowest':>18} {'latest':>18}")
    for name in args.indices:
        key = f"{name}_mean"
        hi, lo, last = max(rows, key=lambda r: r[key]), min(rows, key=lambda r: r[key]), rows[-1]
        cell = lambda r: f"{r[key]:+.2f} {r['date']}"
        print(f"{name:6} {cell(hi):>18} {cell(lo):>18} {cell(last):>18}")
    print(f"\nSaved indices.csv, indices_timeseries.png and indices_map.png in ./{args.out}/")


if __name__ == "__main__":
    main()
