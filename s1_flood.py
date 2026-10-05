"""
Step 11b: map the 2022 Sindh floods with Sentinel-1 radar.

Calm open water reflects the radar pulse away like a mirror, so almost nothing comes back: water
is very dark in VV. Radar sees through the monsoon cloud that hid the floods from optical
satellites. Method:

    1. VV backscatter (dB) on every pass of one orbit, May to December 2022, at 20 m
    2. a 3 x 3 median filter against speckle
    3. a water threshold chosen by Otsu's method on the flood-peak image (the dB value that best
       splits its histogram into dark water and brighter land)
    4. flood = below the threshold AND at least --drop dB darker than the pre-monsoon (June) reference.
       The second rule matters: dry, smooth desert soil is dark in radar too, and stays dark; only a
       real change to standing water makes a pixel much darker than it was.
    5. checks with optical Sentinel-2 (NDWI > 0 = water) on clear dates: one near the flood peak,
       and one in June to see what the areas that were already dark before the monsoon really are

Default area: 50 x 50 km around Dadu, Johi and Khairpur Nathan Shah (Sindh), which were submerged
in early September 2022 as Manchar Lake overflowed.

    python s1_flood.py

Outputs in output/flood/: flood_map.png, flood_timeseries.png, flood_area.csv, flood_peak.tif
"""
import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.transform import from_origin
from scipy.ndimage import median_filter
from shapely.geometry import Point

from s2_cube import lonlat_polygon
from s2_indices import STAC_URL, search_geometry

PLACES = {"Dadu": (67.775, 26.732), "Johi": (67.614, 26.692), "Khairpur Nathan Shah": (67.734, 27.090),
          "Mehar": (67.820, 27.180)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lon", type=float, default=67.70)
    p.add_argument("--lat", type=float, default=26.90)
    p.add_argument("--size", type=int, default=50_000, help="side of the square in metres")
    p.add_argument("--res", type=int, default=20, help="pixel size in metres (20 m: 4 radar pixels averaged)")
    p.add_argument("--start", default="2022-05-01")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--reference", default="2022-06-01/2022-06-30", help="pre-monsoon period for permanent water")
    p.add_argument("--drop", type=float, default=3, help="dB darker than June needed to count as flooded")
    p.add_argument("--optical-window", default="2022-08-25/2022-09-25", help="where to look for a clear Sentinel-2 date")
    p.add_argument("--optical-before", default="2022-06-01/2022-06-30", help="clear Sentinel-2 date before the monsoon")
    p.add_argument("--cache", default=os.path.join("cache", "flood"))
    p.add_argument("--out", default=os.path.join("output", "flood"))
    return p.parse_args()


def box(args):
    pt = gpd.GeoSeries([Point(args.lon, args.lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    c = pt.to_crs(crs).iloc[0]
    h = args.size / 2
    x0, y0 = round((c.x - h) / args.res) * args.res, round((c.y - h) / args.res) * args.res
    return crs, (x0, y0, x0 + args.size, y0 + args.size)


def best_orbit(items):
    """The orbit (direction + relative orbit) with the most passes; one geometry keeps dates comparable."""
    key = lambda i: (i.properties["sat:orbit_state"], i.properties["sat:relative_orbit"])
    counts = Counter(key(i) for i in items)
    (state, rel), n = counts.most_common(1)[0]
    return [i for i in items if key(i) == (state, rel)], f"{state} {rel}", counts


def load_vv(items, crs, bounds, args):
    ids = np.array([i.id for i in items])
    path = os.path.join(args.cache, f"vv_{args.lon}_{args.lat}_{args.size}_{args.res}_{args.start}_{args.end}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["ids"], ids):
            return z["vv"], pd.to_datetime(z["dates"])
    x0, y0, x1, y1 = bounds
    cube = odc.stac.load(items, bands=["vv"], crs=str(crs), resolution=args.res, x=(x0, x1), y=(y0, y1),
                         groupby="solar_day", resampling="average", chunks={"time": 1, "x": 2500, "y": 2500},
                         fail_on_error=False).compute(num_workers=16)
    vv = cube["vv"].values.astype("float32")
    dates = pd.to_datetime(cube.time.values).normalize()
    os.makedirs(args.cache, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ids=ids, vv=vv, dates=dates.values.astype("datetime64[D]"))
    os.replace(tmp, path)
    return vv, dates


def to_db(vv):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(vv)
    d[~np.isfinite(d)] = np.nan
    return d


def smooth(img):
    """3 x 3 median filter that ignores missing pixels at the edges."""
    filled = np.where(np.isfinite(img), img, np.nanmedian(img))
    out = median_filter(filled, size=3)
    out[~np.isfinite(img)] = np.nan
    return out


def otsu(values, bins=256):
    """The threshold that maximises the between-class variance of a two-class split."""
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins, range=(np.percentile(v, 0.5), np.percentile(v, 99.5)))
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def optical_water(crs, bounds, args, window):
    """NDWI > 0 on the clearest Sentinel-2 date in the window, with the SCL clear mask; None if no clear date."""
    a, b = window.split("/")
    items = search_geometry(lonlat_polygon(crs, bounds), a, b, 20)
    if not items:
        return None, None
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    x0, y0, x1, y1 = bounds
    ds = odc.stac.load([item], bands=["B03", "B08", "SCL"], crs=str(crs), resolution=args.res, x=(x0, x1), y=(y0, y1),
                       resampling={"B03": "average", "B08": "average", "SCL": "nearest"},
                       chunks={"x": 2500, "y": 2500}).compute(num_workers=16).isel(time=0)
    g, n, scl = (ds[k].values.astype("float32") for k in ("B03", "B08", "SCL"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (g - n) / (g + n)  # the +1000 offset only enlarges the denominator, so the sign (NDWI > 0) is unchanged
    clear = np.isin(scl, [4, 5, 6])  # vegetation, bare soil, water
    return np.where(clear, ndwi > 0, np.nan), item


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    crs, bounds = box(args)
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    found = catalog.search(collections=["sentinel-1-rtc"], intersects=lonlat_polygon(crs, bounds),
                           datetime=f"{args.start}/{args.end}").item_collection()
    items, orbit, counts = best_orbit(found)
    items = sorted(items, key=lambda i: i.datetime)
    vv, dates = load_vv(items, crs, bounds, args)
    print(f"{len(found)} Sentinel-1 scenes; using orbit {orbit} ({len(dates)} dates, {dates[0].date()} to {dates[-1].date()})")

    db = np.stack([smooth(to_db(v)) for v in vv])
    valid_share = np.isfinite(db).mean(axis=(1, 2))
    keep = valid_share > 0.9  # passes that cover the whole box
    db, dates = db[keep], dates[keep]

    ref_a, ref_b = (pd.Timestamp(x) for x in args.reference.split("/"))
    ref = (dates >= ref_a) & (dates <= ref_b)
    reference = np.nanmedian(db[ref], axis=0)
    # The flood peak: the date with the darkest box; the threshold comes from its histogram.
    peak = int(np.argmin([np.nanmedian(d) for d in db]))
    thr = otsu(db[peak])
    dark_before = reference < thr          # water, or dry smooth ground: radar alone can't tell them apart
    water = db < thr
    flood = water & (db < reference[None] - args.drop)
    km2 = (args.res / 1000) ** 2
    area = pd.DataFrame({"date": [d.date() for d in dates], "water_km2": (water.sum(axis=(1, 2)) * km2).round(1),
                         "flood_km2": (flood.sum(axis=(1, 2)) * km2).round(1)})
    area.to_csv(os.path.join(args.out, "flood_area.csv"), index=False)
    print(f"Water threshold (Otsu on {dates[peak].date()}): {thr:.1f} dB; dark already in June: "
          f"{dark_before.sum() * km2:.0f} km2 (reference: {ref.sum()} June passes); flood also needs a {args.drop:g} dB drop")
    print(area.to_string(index=False))

    transform = from_origin(bounds[0], bounds[3], args.res, args.res)
    peak_map = np.where(flood[peak], 1, np.where(dark_before & water[peak], 2, 0)).astype("uint8")
    with rasterio.open(os.path.join(args.out, "flood_peak.tif"), "w", driver="GTiff", height=peak_map.shape[0],
                       width=peak_map.shape[1], count=1, dtype="uint8", crs=crs, transform=transform,
                       compress="deflate") as dst:
        dst.write(peak_map, 1)
        dst.write_colormap(1, {0: (240, 236, 226, 255), 1: (44, 123, 182, 255), 2: (107, 107, 107, 255)})

    # Optical check near the peak.
    before, bitem = optical_water(crs, bounds, args, args.optical_before)
    if before is not None:
        ok = np.isfinite(before) & dark_before
        print(f"\nWhat was dark in June? Sentinel-2 {bitem.datetime.date()}: of the {dark_before.sum() * km2:.0f} km2, "
              f"optical calls {np.mean(before[ok].astype(bool)):.0%} water (the rest is dry ground that is dark to radar)")
    opt, item = optical_water(crs, bounds, args, args.optical_window)
    if opt is not None:
        od = pd.Timestamp(item.datetime.date())
        near = int(np.argmin(np.abs((dates - od).days)))
        both = np.isfinite(opt)
        s1, s2 = water[near][both], opt[both].astype(bool)
        print(f"\nOptical check: Sentinel-2 {od.date()} ({item.properties['eo:cloud_cover']:.0f}% cloud, "
              f"{both.mean():.0%} of the box clear) vs radar {dates[near].date()}:")
        print(f"  optical water {s2.mean():.1%} of clear pixels, radar water {s1.mean():.1%}; "
              f"agreement {np.mean(s1 == s2):.1%}; of optical water, radar found {np.mean(s1[s2]):.0%}; "
              f"of radar water, optical confirms {np.mean(s2[s1]):.0%}")
        f1, ok = flood[near][both], ~dark_before[both]
        print(f"  flood pixels (with the drop rule): optical confirms {np.mean(s2[f1]):.0%} as water; "
              f"of optical water outside the June-dark area, the flood map found {np.mean(f1[s2 & ok]):.0%}")

    # Figures.
    x0, y0, x1, y1 = bounds
    ext = (0, (x1 - x0) / 1000, 0, (y1 - y0) / 1000)
    to_box = lambda lon, lat: (gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(reference, cmap="gray", vmin=-25, vmax=0, extent=ext)
    axes[0].set_title(f"Before: June 2022 median VV (dB)")
    axes[1].imshow(db[peak], cmap="gray", vmin=-25, vmax=0, extent=ext)
    axes[1].set_title(f"Flood peak: {dates[peak].date()} VV (dB); water is black")
    cmap = ListedColormap(["#f0ece2", "#2c7bb6", "#6b6b6b"])
    axes[2].imshow(peak_map, cmap=cmap, vmin=-0.5, vmax=2.5, extent=ext, interpolation="nearest")
    axes[2].set_title(f"Flooded on {dates[peak].date()}: {flood[peak].sum() * km2:,.0f} km2")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               (("#2c7bb6", "flooded (new, much darker than June)"), ("#6b6b6b", "dark already in June"))]
    axes[2].legend(handles=handles, loc="lower right")
    for ax in axes:
        for name, (lon, lat) in PLACES.items():
            p = to_box(lon, lat)
            px, py = (p.x - x0) / 1000, (p.y - y0) / 1000
            if 0 <= px <= ext[1] and 0 <= py <= ext[3]:
                ax.plot(px, py, "o", color="#d7191c", ms=5)
                ax.annotate(name, (px, py), xytext=(4, 4), textcoords="offset points", color="#d7191c", fontsize=9,
                            fontweight="bold")
        ax.set_xlabel("km east")
    axes[0].set_ylabel("km north")
    fig.suptitle("2022 floods around Dadu, Sindh, seen by Sentinel-1 radar through the monsoon cloud")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "flood_map.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(pd.to_datetime(area["date"]), area["flood_km2"], "o-", color="#2c7bb6", label="flood water")
    ax.plot(pd.to_datetime(area["date"]), area["water_km2"], "--", color="#08306b", alpha=0.6, label="all water")
    ax.set_ylabel("km2")
    ax.set_title(f"Water in the {args.size // 1000} x {args.size // 1000} km box around Dadu, 2022 (one radar orbit)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "flood_timeseries.png"), dpi=130)
    plt.close(fig)
    print(f"Saved flood_map.png, flood_timeseries.png, flood_area.csv, flood_peak.tif in {args.out}")


if __name__ == "__main__":
    main()
