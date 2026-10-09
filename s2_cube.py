"""
Step 10: a data cube for the whole 10 x 10 km square, and a crop map of every 10 m pixel.

The Step 8 model was trained on field medians; here the same pipeline runs per pixel:

    STAC search -> odc.stac.load (lazy xarray cube, 10 m, EPSG:32643)
    -> SCL cloud mask grown by 2 px, hazy dates (from Step 7) dropped, dips dropped
    -> 5 indices -> regular 5-day series (max gap 30 days) -> the 16 set-B features
    -> the chosen Step 8 model -> class probabilities
    -> 3 x 3 mean of probabilities (majority-style smoothing) -> class map,
       "uncertain" where the top probability is below --min-prob, "no data" where too few clear dates

The square is processed in 250 x 250 blocks (plus a 2-pixel halo for the cloud buffer), and each
block's raw pixels are cached in cache/cube/, so a re-run downloads nothing.

    python s2_cube.py

Outputs in output/cube/: crop_map.tif, confidence.tif, crop_map.png, areas.csv, agreement.csv
"""
import argparse
import json
import os
import time
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import rasterio
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter
from shapely.geometry import box, mapping

from s2_classify import MERGE, choose, fit, hand_features, load, models
from s2_fields import CLOUD_SCL, grow
from s2_indices import CLEAR_SCL, INDICES, search_geometry

BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "SCL"]
INDEX_NAMES = ["NDVI", "EVI", "NDMI", "NDRE", "NDWI"]
CODES = {"no data": 0, "wheat": 1, "other_crop": 2, "not_cropped": 3, "uncertain": 4}
COLOURS = {"no data": "#ffffff", "wheat": "#c9962b", "other_crop": "#2c774c", "not_cropped": "#9a9a9a",
           "uncertain": "#e8dfd0"}
PIXEL_HA = 0.01  # one 10 m pixel = 100 m2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--known", default="fields.geojson", help="the 3 fields the square is centred on")
    p.add_argument("--size", type=int, default=10_000, help="side of the square in metres")
    p.add_argument("--block", type=int, default=250, help="block size in pixels")
    p.add_argument("--start", default="2025-08-15", help="first image (a little before the grid, for interpolation)")
    p.add_argument("--end", default="2026-06-30", help="last image (a little after the features, for interpolation)")
    p.add_argument("--grid-start", default="2025-09-01", help="first date of the 5-day grid, as in Step 6")
    p.add_argument("--grid-end", default="2026-05-31", help="last grid date the features need")
    p.add_argument("--max-cloud", type=float, default=50)
    p.add_argument("--max-gap", type=int, default=30)
    p.add_argument("--dip", type=float, default=0.1)
    p.add_argument("--dip-days", type=int, default=20)
    p.add_argument("--cloud-buffer", type=int, default=2)
    p.add_argument("--min-prob", type=float, default=0.6, help="below this top probability a pixel is 'uncertain'")
    p.add_argument("--rgb-date", default="2026-03-07", help="clear date for the true-colour zoom (wheat at its peak)")
    p.add_argument("--zoom", type=float, default=2.5, help="side of the zoomed view in km")
    p.add_argument("--min-clear", type=int, default=8, help="fewer clear Oct-May dates than this is 'no data'")
    p.add_argument("--obs", default=os.path.join("output", "labels", "fields.csv"), help="Step 7 observations (hazy dates)")
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--matrix", default=os.path.join("output", "labels", "series_matrix.csv"))
    p.add_argument("--classify", default=os.path.join("output", "classify"))
    p.add_argument("--cache", default=os.path.join("cache", "cube"))
    p.add_argument("--prefetch", type=int, metavar="N",
                   help="only download up to N blocks that are not cached yet, then stop (for short sessions)")
    p.add_argument("--out", default=os.path.join("output", "cube"))
    return p.parse_args(argv)


# ---------- the square and the cube ----------

def square(args):
    """The same 10 x 10 km square as s2_sample.py, snapped to the 20 m grid so 20 m pixels line up."""
    known = gpd.read_file(args.known)
    crs = known.estimate_utm_crs()
    centre = known.to_crs(crs).union_all().centroid
    x0 = round((centre.x - args.size / 2) / 20) * 20
    y0 = round((centre.y - args.size / 2) / 20) * 20
    return crs, (x0, y0, x0 + args.size, y0 + args.size)


def lonlat_polygon(crs, bounds):
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = bounds
    ring = [to_ll.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return {"type": "Polygon", "coordinates": [ring]}


def lazy_cube(items, crs, bounds, chunk=256):
    """Nothing is read here: odc-stac builds a dask-backed cube on one 10 m grid."""
    x0, y0, x1, y1 = bounds
    return odc.stac.load([planetary_computer.sign(i) for i in items], bands=BANDS, crs=str(crs), resolution=10,
                         x=(x0, x1), y=(y0, y1), groupby="solar_day", resampling="nearest",
                         chunks={"time": 1, "x": chunk, "y": chunk}, fail_on_error=False)


def block_path(rows, cols, args):
    return os.path.join(args.cache, f"r{rows.start}_c{cols.start}_{args.block}_{args.start}_{args.end}.npz")


def load_block(items, crs, bounds, rows, cols, halo, args):
    """Raw uint16 pixels of one block plus a halo, cached by block and the exact list of scenes."""
    path = block_path(rows, cols, args)
    ids = np.array([i.id for i in items])
    if os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["ids"], ids):
            return {b: z[b] for b in BANDS}, z["days"]
    # Load exactly this block (with its halo) as one chunk per band and date: one read each.
    n = (bounds[2] - bounds[0]) // 10
    r0, r1 = max(rows.start - halo, 0), min(rows.stop + halo, n)
    c0, c1 = max(cols.start - halo, 0), min(cols.stop + halo, n)
    x0, y1 = bounds[0], bounds[3]
    part_bounds = (x0 + c0 * 10, y1 - r1 * 10, x0 + c1 * 10, y1 - r0 * 10)
    cube = lazy_cube(items, crs, part_bounds, chunk=max(r1 - r0, c1 - c0))
    part = cube.compute(num_workers=16)  # reading is network-bound, so more threads than cores
    assert part.sizes["y"] == r1 - r0 and part.sizes["x"] == c1 - c0, "block grid does not line up"
    assert part.y.values[0] > part.y.values[-1], "expected rows from north to south"
    days = np.array([(pd.Timestamp(t).date() - date(2025, 1, 1)).days for t in part.time.values])
    raw = {b: np.nan_to_num(part[b].values, nan=0).astype("uint16") for b in BANDS}  # failed reads -> 0 = no data
    os.makedirs(args.cache, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ids=ids, days=days, **raw)
    os.replace(tmp, path)
    return raw, days


# ---------- per-pixel series ----------

def last_and_next(valid):
    """For every time t and pixel: the index of the last valid observation at or before t (-1 if none)
    and of the next valid one at or after t (T if none)."""
    t = np.arange(valid.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(valid, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(valid, t, valid.shape[0])[::-1], axis=0)[::-1]
    return last, nxt


def take(values, idx):
    """values[idx[p], p] for every pixel p, NaN where idx is out of range."""
    T = values.shape[0]
    safe = np.clip(idx, 0, T - 1)
    if idx.ndim == 1:
        out = values[safe, np.arange(values.shape[1])]
    else:
        out = np.take_along_axis(values, safe, axis=0)
    return np.where((idx >= 0) & (idx < T), out, np.nan)


def dips(ndvi, valid, days, args):
    """The Step 5 dip rule per pixel: more than --dip below both neighbouring clear observations,
    each within --dip-days."""
    last, nxt = last_and_next(valid)
    T = ndvi.shape[0]
    prev = np.vstack([np.full((1, ndvi.shape[1]), -1), last[:-1]])  # last valid strictly before t
    after = np.vstack([nxt[1:], np.full((1, ndvi.shape[1]), T)])     # next valid strictly after t
    d = days[:, None]
    dprev = d - np.where(prev >= 0, days[np.clip(prev, 0, T - 1)], -10_000)
    dnext = np.where(after < T, days[np.clip(after, 0, T - 1)], 10_000) - d
    vp, vn = take(ndvi, prev), take(ndvi, after)
    with np.errstate(invalid="ignore"):
        return valid & (dprev <= args.dip_days) & (dnext <= args.dip_days) & (ndvi < vp - args.dip) & (ndvi < vn - args.dip)


def regularise(values, valid, days, grid_days, max_gap):
    """Step 6's regularise for every pixel at once: linear interpolation between the clear observations
    either side of each grid date; empty outside them or across a gap longer than max_gap."""
    last, nxt = last_and_next(valid)
    T = values.shape[0]
    out = np.full((len(grid_days), values.shape[1]), np.nan, dtype="float32")
    for g, gd in enumerate(grid_days):
        k = np.searchsorted(days, gd, side="right") - 1    # last observation date on or before gd
        k2 = np.searchsorted(days, gd, side="left")        # first observation date on or after gd
        left = last[k] if k >= 0 else np.full(values.shape[1], -1)
        right = nxt[k2] if k2 < T else np.full(values.shape[1], T)
        ok = (left >= 0) & (right < T)
        dl = np.where(ok, days[np.clip(left, 0, T - 1)], 0)
        dr = np.where(ok, days[np.clip(right, 0, T - 1)], 0)
        vl, vr = take(values, left), take(values, right)
        exact = ok & (dl == gd)
        w = np.where(dr > dl, (gd - dl) / np.maximum(dr - dl, 1), 0.0)
        v = np.where(exact, vl, vl + (vr - vl) * w)
        v[~ok | ((dr - dl > max_gap) & ~exact)] = np.nan
        out[g] = v
    return out


def block_features(raw, days, hazy_days, grid, cargs, args, with_series=False):
    """Raw pixels of one block -> the 16 set-B features per pixel, and the clear Oct-May date count
    (plus, with with_series, the regular 5-day series of every index as a pixels x columns table)."""
    T, h, w = raw["SCL"].shape
    scl = raw["SCL"]
    cloud = np.stack([grow(np.isin(s, CLOUD_SCL), args.cloud_buffer) for s in scl])
    clear = np.isin(scl, CLEAR_SCL) & ~cloud
    refl = {b: np.clip((raw[b].astype("float32") - 1000) / 10000, 1e-4, None) for b in BANDS if b != "SCL"}
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = {n: INDICES[n]["formula"](refl).reshape(T, -1).astype("float32") for n in INDEX_NAMES}
    clear = clear.reshape(T, -1) & np.isfinite(idx["NDVI"])
    dipped = dips(idx["NDVI"], clear, days, args)           # neighbours include hazy dates, as in Step 5
    valid = clear & ~dipped & ~np.isin(days, hazy_days)[:, None]
    grid_days = np.array([(d.date() - date(2025, 1, 1)).days for d in grid])
    cols = {}
    for n in INDEX_NAMES:
        v = np.where(valid & np.isfinite(idx[n]), idx[n], np.nan)
        series = regularise(v, valid & np.isfinite(idx[n]), days, grid_days, args.max_gap)
        for g, d in enumerate(grid):
            cols[f"{n}_{d.date()}"] = series[g]
    matrix = pd.DataFrame(cols)
    feats = hand_features(matrix, cargs)
    window = (days >= (date(2025, 10, 1) - date(2025, 1, 1)).days) & (days <= (date(2026, 5, 31) - date(2025, 1, 1)).days)
    n_clear = valid[window].sum(axis=0)
    return (feats, n_clear, matrix) if with_series else (feats, n_clear)


# ---------- model, map and outputs ----------

def final_model(args):
    """Refit the Step 8 choice (same data, same rule) on all training fields."""
    cargs = argparse.Namespace(fields=args.fields, matrix=args.matrix, season_start="2025-10-01",
                               season_end="2026-05-31", max_missing=0.5, binary=False)
    matrix, y_all, everyone = load(cargs)
    props = everyone.loc[y_all.index]
    is_train = (props["role"] == "train").values
    y, groups = y_all[is_train], props.loc[is_train, "block"]
    feats = hand_features(matrix, cargs).loc[y_all.index]
    results = pd.read_csv(os.path.join(args.classify, "cv_results.csv"))
    best, *_ = choose(results, len(y), {"A": 200, "B": feats.shape[1]})
    if best["features"] != "B":
        raise SystemExit("The chosen Step 8 model uses feature set A; this script computes set B per pixel.")
    model = fit(models()[best["model"]], feats[is_train], y, groups)
    return model, best["model"], cargs, list(feats.columns)


def write_tif(path, array, crs, transform, nodata=None, colormap=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=array.dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(array, 1)
        if colormap:
            dst.write_colormap(1, colormap)


def plot_map(classes, conf, fields, transform, bounds, path):
    names = list(CODES)
    cmap = ListedColormap([COLOURS[n] for n in names])
    x0, y0, x1, y1 = bounds
    extent = (0, (x1 - x0) / 1000, 0, (y1 - y0) / 1000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    axes[0].imshow(classes, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, extent=extent, interpolation="nearest")
    im = axes[1].imshow(conf, cmap="viridis", vmin=0.33, vmax=1, extent=extent, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="top class probability")
    for ax in axes:
        for geom in fields.geometry:
            xs, ys = geom.exterior.xy
            ax.plot((np.array(xs) - x0) / 1000, (np.array(ys) - y0) / 1000, color="black", lw=0.5)
        ax.set_xlabel("km east")
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
    axes[0].set_ylabel("km north")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[n], ec="black", lw=0.3, label=n.replace("_", " "))
               for n in names if n != "no data"]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    axes[0].set_title("Rabi 2026 crop map (10 m pixels, 3x3 smoothed); black: the 100 labelled fields")
    axes[1].set_title("Model confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zoom(rgb, classes, known, bounds, args, path):
    """True colour next to the map for a small window around the 3 known fields."""
    x0, y0, x1, y1 = bounds
    c = known.union_all().centroid
    half = args.zoom * 500
    col0 = int((c.x - half - x0) // 10)
    row0 = int((y1 - c.y - half) // 10)
    size = int(args.zoom * 100)
    rows, cols = slice(max(row0, 0), row0 + size), slice(max(col0, 0), col0 + size)
    img = np.clip((rgb[:, rows, cols].astype("float32") - 1000) / 10000 / 0.2, 0, 1).transpose(1, 2, 0)
    names = list(CODES)
    cmap = ListedColormap([COLOURS[n] for n in names])
    ext = (cols.start * 10 / 1000, cols.stop * 10 / 1000, (1000 - rows.stop) * 10 / 1000, (1000 - rows.start) * 10 / 1000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    axes[0].imshow(img, extent=ext)
    axes[0].set_title(f"Sentinel-2 true colour, {args.rgb_date}")
    axes[1].imshow(classes[rows, cols], cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, extent=ext, interpolation="nearest")
    axes[1].set_title("Crop map, Rabi 2026")
    for ax in axes:
        for geom in known.geometry:
            xs, ys = geom.exterior.xy
            ax.plot((np.array(xs) - x0) / 1000, (np.array(ys) - y0) / 1000, color="white" if ax is axes[0] else "black", lw=1)
        ax.set_xlabel("km east")
    axes[0].set_ylabel("km north")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[n], ec="black", lw=0.3, label=n.replace("_", " "))
               for n in names if n != "no data"]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    crs, bounds = square(args)
    items = search_geometry(lonlat_polygon(crs, bounds), args.start, args.end, args.max_cloud)
    if any(i.properties.get("s2:processing_baseline", "00.00") < "04.00" for i in items):
        raise SystemExit("A scene older than processing baseline 04.00: the fixed 1000 offset would be wrong")
    obs = pd.read_csv(args.obs, parse_dates=["date"])
    hazy = sorted(obs.loc[obs["flag"].str.contains("haze"), "date"].dt.date.unique())
    hazy_days = np.array([(d - date(2025, 1, 1)).days for d in hazy])
    grid = pd.date_range(args.grid_start, args.grid_end, freq="5D")
    n = args.size // 10
    blocks = [(slice(r, min(r + args.block, n)), slice(c, min(c + args.block, n)))
              for r in range(0, n, args.block) for c in range(0, n, args.block)]
    if args.prefetch is not None:
        todo = [b for b in blocks if not os.path.exists(block_path(*b, args))]
        for rows, cols in todo[:args.prefetch]:
            load_block(items, crs, bounds, rows, cols, args.cloud_buffer, args)
            print(f"  downloaded block rows {rows.start}-{rows.stop}, cols {cols.start}-{cols.stop}, "
                  f"{time.time() - t0:.0f} s", flush=True)
        print(f"{len(todo[args.prefetch:])} of {len(blocks)} blocks still to download")
        return
    model, name, cargs, feature_names = final_model(args)
    print(f"{len(items)} scenes {items[0].datetime.date()} to {items[-1].datetime.date()}; "
          f"{len(hazy)} hazy dates dropped; model: set B + {name}")

    probs = np.full((len(model.classes_), n, n), np.nan, dtype="float32")
    n_clear = np.zeros((n, n), dtype="int16")
    feats_all = np.full((len(feature_names), n, n), np.nan, dtype="float32")
    rgb = np.zeros((3, n, n), dtype="uint16")
    rgb_day = (date.fromisoformat(args.rgb_date) - date(2025, 1, 1)).days
    halo = args.cloud_buffer
    for r in range(0, n, args.block):
        for c in range(0, n, args.block):
            rows, cols = slice(r, min(r + args.block, n)), slice(c, min(c + args.block, n))
            raw, days = load_block(items, crs, bounds, rows, cols, halo, args)
            # cut the halo off after the cloud buffer has used it
            top, left = rows.start - max(rows.start - halo, 0), cols.start - max(cols.start - halo, 0)
            h, w = rows.stop - rows.start, cols.stop - cols.start
            feats, nc = block_features(raw, days, hazy_days, grid, cargs, args)
            H, W = raw["SCL"].shape[1:]
            keep = np.zeros((H, W), bool)
            keep[top:top + h, left:left + w] = True
            keep = keep.ravel()
            f = feats[keep][feature_names]
            probs[:, rows, cols] = model.predict_proba(f).T.reshape(-1, h, w)
            feats_all[:, rows, cols] = f.to_numpy().T.reshape(-1, h, w)
            n_clear[rows, cols] = nc[keep].reshape(h, w)
            if rgb_day in days:
                t = int(np.flatnonzero(days == rgb_day)[0])
                for i, b in enumerate(("B04", "B03", "B02")):
                    rgb[i, rows, cols] = raw[b][t, top:top + h, left:left + w]
            print(f"  block rows {r}-{rows.stop}, cols {c}-{cols.stop}: {len(days)} dates, "
                  f"{time.time() - t0:.0f} s", flush=True)

    smooth = uniform_filter(probs, size=(1, 3, 3), mode="nearest")
    top_p = smooth.max(axis=0)
    best = np.array(model.classes_)[smooth.argmax(axis=0)]
    classes = np.vectorize(CODES.get)(best).astype("uint8")
    classes[top_p < args.min_prob] = CODES["uncertain"]
    classes[n_clear < args.min_clear] = CODES["no data"]
    raw_classes = np.array(model.classes_)[probs.argmax(axis=0)]

    transform = from_origin(bounds[0], bounds[3], 10, 10)
    cmap = {v: tuple(int(COLOURS[k][i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k, v in CODES.items()}
    write_tif(os.path.join(args.out, "crop_map.tif"), classes, crs, transform, nodata=0, colormap=cmap)
    write_tif(os.path.join(args.out, "confidence.tif"), (top_p * 100).round().astype("uint8"), crs, transform)
    np.save(os.path.join(args.cache, "features.npy"), feats_all)

    total = classes.size
    areas = pd.DataFrame([{"class": k, "pixels": int((classes == v).sum()),
                           "hectares": round(int((classes == v).sum()) * PIXEL_HA, 1),
                           "share": round((classes == v).sum() / total, 3)} for k, v in CODES.items()])
    areas.to_csv(os.path.join(args.out, "areas.csv"), index=False)
    changed = np.mean(best != raw_classes)

    # How the map agrees with the 100 labelled fields: the majority class of the pixels inside each outline.
    fields = gpd.read_file(args.fields).to_crs(crs)
    rows = []
    inv = {v: k for k, v in CODES.items()}
    for _, fld in fields.iterrows():
        inside = ~geometry_mask([mapping(fld.geometry)], out_shape=classes.shape, transform=transform)
        vals = classes[inside]
        major = inv[np.bincount(vals, minlength=5).argmax()] if vals.size else "outside"
        label = MERGE.get(fld["rabi_2026"], "unknown")
        rows.append({"field": fld["name"], "role": fld["role"], "label": label, "map": major,
                     "pixels": int(vals.size), "same_class_share": round(float(np.mean(vals == CODES.get(major, -1))), 2)
                     if vals.size else 0})
    agree = pd.DataFrame(rows)
    agree.to_csv(os.path.join(args.out, "agreement.csv"), index=False)
    plot_map(classes, top_p, fields, transform, bounds, os.path.join(args.out, "crop_map.png"))
    plot_zoom(rgb, classes, gpd.read_file(args.known).to_crs(crs), bounds, args, os.path.join(args.out, "crop_map_zoom.png"))

    print(f"\nArea of the {args.size // 1000} x {args.size // 1000} km square ({total * PIXEL_HA:.0f} ha):")
    for _, r in areas.iterrows():
        print(f"  {r['class']:<12} {r['hectares']:>8.1f} ha  {r['share']:>6.1%}")
    print(f"Smoothing changed {changed:.1%} of pixels; clear Oct-May dates per pixel: "
          f"median {np.median(n_clear):.0f}, min {n_clear.min()}")
    known = agree[agree["label"] != "unknown"]
    for role in ("train", "test"):
        k = known[known["role"] == role]
        print(f"Map vs {role} labels (field majority): {np.mean(k['label'] == k['map']):.0%} of {len(k)}")
    print(f"Saved crop_map.tif, confidence.tif, crop_map.png, areas.csv, agreement.csv in {args.out} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
