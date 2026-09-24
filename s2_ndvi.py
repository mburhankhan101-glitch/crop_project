"""
Crop health (NDVI) over time for a patch of farmland, from free Sentinel-2
satellite images hosted on Microsoft Planetary Computer. No account needed.

Default spot: farmland between Raiwind and Kasur, south-west of Lahore.

    python s2_ndvi.py                            # default Lahore farmland
    python s2_ndvi.py --lat 31.30 --lon 74.07    # any other spot
    python s2_ndvi.py --start 2024-01-01 --end 2024-12-31

Writes to ./output/:
    ndvi.csv             one row per clear satellite pass
    ndvi_timeseries.png  crop health over the season
    ndvi_map.png         true-colour image next to an NDVI map of the area
"""
import argparse
import csv
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
PIXEL_M = 10  # Sentinel-2 red, green, blue and near-infrared bands are 10 m per pixel

# Scene Classification Layer (SCL) values we trust: 4 vegetation, 5 bare soil, 6 water.
# The rest are cloud, cloud shadow, thin cirrus, snow or missing data.
CLEAR_SCL = [4, 5, 6]

# Planetary Computer supports the cloud-cover "query" filter but doesn't advertise it.
warnings.filterwarnings("ignore", message=".*does not conform to QUERY")


def parse_args():
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=31.1923, help="latitude of the centre (default: farmland SW of Lahore)")
    p.add_argument("--lon", type=float, default=74.3019, help="longitude of the centre")
    p.add_argument("--size", type=int, default=400, help="side of the square to average, in metres (default 400)")
    p.add_argument("--map-size", type=int, default=2000, help="side of the square drawn in ndvi_map.png, in metres")
    p.add_argument("--start", default=str(today - timedelta(days=395)), help="YYYY-MM-DD (default: ~13 months ago)")
    p.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    p.add_argument("--max-cloud", type=float, default=50, help="skip scenes with more cloud than this %% overall")
    p.add_argument("--min-clear", type=float, default=0.8, help="fraction of the square that must be cloud-free")
    p.add_argument("--out", default="output", help="output folder")
    return p.parse_args()


def search(lat, lon, start, end, max_cloud):
    """Find Sentinel-2 scenes over the point, keeping the least cloudy one per day."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
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
    """Read a size_m x size_m square centred on (lat, lon) for each band, on the 10 m grid.

    The first band must be a 10 m band; coarser bands (like the 20 m SCL) are resampled to match.
    Returns None if the square runs off the edge of this tile.
    """
    crs = item.properties.get("proj:code") or f"EPSG:{item.properties['proj:epsg']}"
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    x, y, half = xs[0], ys[0], size_m / 2
    expected = (round(size_m / PIXEL_M),) * 2

    out, shape = {}, None
    for band in bands:
        with rasterio.open(item.assets[band].href) as src:
            win = from_bounds(x - half, y - half, x + half, y + half, src.transform)
            win = win.round_offsets().round_lengths()
            if shape is None:
                arr = src.read(1, window=win)
                if arr.shape != expected:
                    return None
                shape = arr.shape
            else:
                arr = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
        out[band] = arr
    return out


def reflectance(dn, item):
    """Convert raw digital numbers to surface reflectance (0-1).

    Since processing baseline 04.00 (January 2022) ESA adds 1000 to every value,
    which must be removed first or NDVI comes out wrong.
    """
    offset = 1000 if item.properties.get("s2:processing_baseline", "00.00") >= "04.00" else 0
    return np.clip((dn.astype("float32") - offset) / 10000, 1e-4, None)


def ndvi(red, nir):
    # Healthy plants absorb red light and strongly reflect near-infrared.
    return (nir - red) / (nir + red)


def scene_ndvi(item, args):
    """Mean NDVI of the cloud-free pixels in the square, or None if too cloudy."""
    try:
        sq = read_square(item, args.lat, args.lon, args.size, ["B04", "B08", "SCL"])
    except rasterio.errors.RasterioIOError as e:
        print(f"  {item.datetime.date()}: could not read ({e})")
        return None
    if sq is None:
        return None
    clear = np.isin(sq["SCL"], CLEAR_SCL) & (sq["B04"] > 0) & (sq["B08"] > 0)
    if clear.mean() < args.min_clear:
        return None
    values = ndvi(reflectance(sq["B04"], item), reflectance(sq["B08"], item))[clear]
    return {
        "date": item.datetime.date(),
        "ndvi_mean": float(values.mean()),
        "ndvi_std": float(values.std()),
        "clear_fraction": float(clear.mean()),
        "scene_cloud_pct": float(item.properties["eo:cloud_cover"]),
        "item": item,
    }


def plot_timeseries(rows, args, path):
    dates = [r["date"] for r in rows]
    mean = np.array([r["ndvi_mean"] for r in rows])
    std = np.array([r["ndvi_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Punjab's two cropping seasons: Rabi (winter, mainly wheat) and Kharif (summer, rice/cotton/maize).
    for year in range(dates[0].year - 1, dates[-1].year + 1):
        ax.axvspan(date(year, 11, 1), date(year + 1, 4, 30), color="#f2c14e", alpha=0.12, lw=0)
        ax.axvspan(date(year + 1, 5, 1), date(year + 1, 10, 31), color="#4e9af2", alpha=0.10, lw=0)

    ax.fill_between(dates, mean - std, mean + std, color="#2e7d32", alpha=0.15, lw=0, label="spread across the square")
    ax.plot(dates, mean, "-o", color="#2e7d32", ms=4, lw=1.8, label="mean NDVI")

    ax.set_xlim(dates[0] - timedelta(days=7), dates[-1] + timedelta(days=7))
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0.2, color="grey", lw=0.8, ls=":")
    ax.text(dates[0], 0.21, " bare soil / harvested below this", fontsize=8, color="grey", va="bottom")
    ax.axhline(0.6, color="grey", lw=0.8, ls=":")
    ax.text(dates[0], 0.61, " dense healthy crop above this", fontsize=8, color="grey", va="bottom")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("NDVI (crop greenness)")
    ax.set_title(
        f"Crop health at {args.lat:.4f}N, {args.lon:.4f}E  ·  {args.size} m square  ·  "
        f"{len(rows)} clear Sentinel-2 passes"
    )
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(color="#f2c14e", alpha=0.35, label="Rabi season (Nov–Apr, wheat)"),
        mpatches.Patch(color="#4e9af2", alpha=0.3, label="Kharif season (May–Oct, rice etc.)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(item, args, path):
    """True-colour image beside an NDVI map, with the averaged square outlined."""
    sq = read_square(item, args.lat, args.lon, args.map_size, ["B04", "B03", "B02", "B08", "SCL"])
    if sq is None:
        print("  map area runs off the tile edge; skipping ndvi_map.png")
        return
    red, green, blue, nir = (reflectance(sq[b], item) for b in ("B04", "B03", "B02", "B08"))
    rgb = np.clip(np.dstack([red, green, blue]) * 3.5, 0, 1)  # brighten; raw reflectance looks very dark
    ndvi_map = np.where(np.isin(sq["SCL"], CLEAR_SCL), ndvi(red, nir), np.nan)

    n = rgb.shape[0]
    box = args.size / PIXEL_M
    corner = (n - box) / 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(rgb)
    axes[0].set_title("True colour (what your eye would see)")
    im = axes[1].imshow(ndvi_map, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    axes[1].set_title("NDVI (red = bare / stressed, green = healthy crop)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.add_patch(mpatches.Rectangle((corner, corner), box, box, fill=False, ec="white", lw=1.5, ls="--"))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sentinel-2, {item.datetime.date()}  ·  {args.map_size / 1000:g} km across  ·  dashed box = averaged square")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Searching Sentinel-2 scenes over {args.lat}, {args.lon} from {args.start} to {args.end} ...")
    items = search(args.lat, args.lon, args.start, args.end, args.max_cloud)
    print(f"Found {len(items)} passes with under {args.max_cloud:g}% cloud. Reading pixels ...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(lambda it: scene_ndvi(it, args), items) if r is not None]
    rows.sort(key=lambda r: r["date"])
    print(f"{len(rows)} passes were clear over your square.")
    if not rows:
        print("Nothing to plot. Try a longer date range or a higher --max-cloud.")
        return

    csv_path = os.path.join(args.out, "ndvi.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ndvi_mean", "ndvi_std", "clear_fraction", "scene_cloud_pct"])
        for r in rows:
            w.writerow([r["date"], f"{r['ndvi_mean']:.4f}", f"{r['ndvi_std']:.4f}",
                        f"{r['clear_fraction']:.2f}", f"{r['scene_cloud_pct']:.1f}"])

    plot_timeseries(rows, args, os.path.join(args.out, "ndvi_timeseries.png"))
    plot_map(rows[-1]["item"], args, os.path.join(args.out, "ndvi_map.png"))

    peak = max(rows, key=lambda r: r["ndvi_mean"])
    low = min(rows, key=lambda r: r["ndvi_mean"])
    print(f"\nPeak greenness: {peak['ndvi_mean']:.2f} on {peak['date']}")
    print(f"Lowest:         {low['ndvi_mean']:.2f} on {low['date']}")
    print(f"Latest:         {rows[-1]['ndvi_mean']:.2f} on {rows[-1]['date']}")
    print(f"\nSaved {csv_path}, ndvi_timeseries.png and ndvi_map.png in ./{args.out}/")


if __name__ == "__main__":
    main()
