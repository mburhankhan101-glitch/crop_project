"""
Semi-automatic field boundaries: grows a field outwards from each sample point over pixels
whose NDVI through the year looks like the point's own, then turns it into a polygon.

Why the whole year: neighbouring fields can look identical in one season and completely
different in another (the three fields near Raiwind were one rice block in September and
three different crops in March), so pixels are compared on a monthly NDVI profile.

For each point:
    reference   the median monthly NDVI profile of the 3 x 3 pixels at the point
    distance    for each nearby pixel, the mean absolute difference from that profile, over
                the months both are cloud-free
    field       the pixels under --threshold that connect to the point, after cutting
                1-pixel bridges to neighbouring fields and filling holes

The threshold is calibrated on fields whose boundaries are known (--calibrate), by the
overlap between the grown field and the real one (intersection over union, IoU).

    python s2_delineate.py --calibrate fields.geojson     # check thresholds on known fields
    python s2_delineate.py                                # grow a field at every sample point

Writes to ./labels/:
    fields_auto.geojson   one polygon per point with a quality note, and empty label properties
    review_01.png ...     every field outlined on false colour in March and September
"""
import argparse
import json
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import Point, box, mapping, shape
from shapely.geometry.polygon import orient

from s2_fields import item_crs, read_bounds
from s2_indices import CLEAR_SCL, reflectance, search_geometry

BANDS = ["B03", "B04", "B08", "SCL"]
PIXEL_M = 10
SOLIDITY = 0.8           # below this a grown field is treated as ragged, probably several fields merged
RETRY = [0.06, 0.05, 0.04]  # stricter thresholds tried when a field comes out ragged or too large
MIN_ACRES = 0.2          # smaller results usually mean the point sits on a field edge
LABEL_KEYS = ["rabi_2026", "rabi_2026_source", "rabi_2026_confidence",
              "kharif_2026", "kharif_2026_source", "kharif_2026_confidence", "notes"]


def parse_args():
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", default=os.path.join("labels", "sample_points.geojson"))
    p.add_argument("--calibrate", metavar="GEOJSON", help="grow fields at these known fields and report IoU per threshold")
    p.add_argument("--threshold", type=float, default=0.08,
                   help="max mean NDVI difference from the point's profile (default 0.08, from calibration)")
    p.add_argument("--radius", type=int, default=300, help="search at most this far from the point, in metres")
    p.add_argument("--flags", default=os.path.join("output", "fields.csv"),
                   help="s2_fields.py output; dates flagged for every field are skipped (default output/fields.csv)")
    p.add_argument("--start", default=str(today - timedelta(days=395)), help="YYYY-MM-DD (default: ~13 months ago)")
    p.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    p.add_argument("--out", default="labels", help="output folder")
    return p.parse_args()


def monthly_stack(area_utm, crs, args):
    """The least cloudy scene of each month over the area, as NDVI with clouds set to NaN.

    Returns (dates, ndvi [months, rows, cols], transform, per-month arrays of raw bands).
    """
    area_ll = gpd.GeoSeries([area_utm], crs=crs).to_crs("EPSG:4326").iloc[0]
    items = search_geometry(mapping(area_ll), args.start, args.end, max_cloud=60)

    # A scene can be cloud-free and still hazy (20 Feb 2026). Reuse s2_fields.py's flags: a date on
    # which every field was flagged is an atmosphere problem, so it can't be a month's best scene.
    bad = set()
    if os.path.exists(args.flags):
        import pandas as pd
        flags = pd.read_csv(args.flags)
        # Images judged by eye (label sheets) are stricter: one flagged field is enough to skip a date.
        rule = (lambda f: (f != "ok").any()) if getattr(args, "skip_any_flagged", False) else (lambda f: (f != "ok").all())
        per_date = flags.groupby("date")["flag"].agg(rule)
        bad = {date.fromisoformat(d) for d, flagged in per_date.items() if flagged}
        print(f"Skipping {len(bad)} dates flagged as hazy or cloudy in {args.flags}: "
              + ", ".join(f"{d:%d %b %y}" for d in sorted(bad)))
    items = [it for it in items if it.datetime.date() not in bad]
    best = {}
    for it in items:
        key = (it.datetime.year, it.datetime.month)
        if key not in best or it.properties["eo:cloud_cover"] < best[key].properties["eo:cloud_cover"]:
            best[key] = it
    items = [best[k] for k in sorted(best)]
    print(f"Reading {len(items)} monthly scenes over {area_utm.bounds[2] - area_utm.bounds[0]:.0f} m square "
          f"(cached after the first run) ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        reads = list(pool.map(lambda it: read_bounds(it, area_utm.bounds, BANDS), items))

    dates, ndvi, raw, transform = [], [], [], None
    for it, read in zip(items, reads):
        if read is None:
            continue
        arrays, transform = read
        red, nir = reflectance(arrays["B04"], it), reflectance(arrays["B08"], it)
        clear = np.isin(arrays["SCL"], CLEAR_SCL) & (arrays["B04"] > 0)
        ndvi.append(np.where(clear, (nir - red) / (nir + red), np.nan).astype("float32"))
        raw.append((it, arrays, clear))
        dates.append(it.datetime.date())
    return dates, np.stack(ndvi), transform, raw


def component_at(mask, r, c):
    """The 4-connected region of mask containing pixel (r, c), or None."""
    if not mask[r, c]:
        return None
    labels, _ = ndimage.label(mask)
    return labels == labels[r, c]


def grow_field(ndvi, transform, x, y, threshold, radius_px, min_months=6):
    """Grow a field from the point (x, y). Returns (polygon or None, quality notes, reference profile)."""
    months, rows, cols = ndvi.shape
    col, row = (int(v) for v in ~transform * (x, y))
    r0, r1 = max(row - radius_px, 0), min(row + radius_px + 1, rows)
    c0, c1 = max(col - radius_px, 0), min(col + radius_px + 1, cols)
    local = ndvi[:, r0:r1, c0:c1]
    sr, sc = row - r0, col - c0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-cloudy months give NaN medians
        ref = np.nanmedian(local[:, max(sr - 1, 0):sr + 2, max(sc - 1, 0):sc + 2].reshape(months, -1), axis=1)
    diff = np.abs(local - ref[:, None, None])
    n = (~np.isnan(diff)).sum(axis=0)
    dist = np.where(n >= min_months, np.nansum(diff, axis=0) / np.maximum(n, 1), np.inf)

    similar = dist < threshold
    # Cut 1-pixel bridges (a shared field ridge that happens to look similar) between fields.
    region = component_at(ndimage.binary_opening(similar, structure=np.ones((3, 3))), sr, sc)
    notes = []
    if region is None:
        region = component_at(similar, sr, sc)
    if region is None:
        return None, ["no field found at the point (on a boundary?)"], ref
    region = ndimage.binary_fill_holes(region)

    local_transform = transform * Affine.translation(c0, r0)
    polys = [shape(g) for g, v in rasterio.features.shapes(region.astype("uint8"), mask=region,
                                                           transform=local_transform) if v == 1]
    poly = orient(max(polys, key=lambda p: p.area).simplify(PIXEL_M / 2, preserve_topology=True), 1.0)

    acres = poly.area / 4047
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        notes.append("large: may include neighbouring fields")
    if acres < 1:
        notes.append("small")
    # Solidity = area / convex hull area. Fields, long strips included, are nearly convex (~0.9);
    # several fields merged through their corners are ragged and fill much less of their hull.
    if poly.area / poly.convex_hull.area < SOLIDITY:
        notes.append("irregular")
    if np.nanmax(ref) < 0.3:
        notes.append("never green: non-crop?")
    if n[sr, sc] < min_months:
        notes.append("too cloudy")
    return poly, notes, ref


def grow_with_retry(ndvi, transform, x, y, args):
    """Grow a field; if it comes out ragged or too large, retry with stricter thresholds.

    Neighbouring fields with the same crop history can merge into one ragged blob, so the first
    stricter result that looks like a single field is kept. Returns (polygon, notes, threshold).
    """
    best = None
    for t in [args.threshold] + [r for r in RETRY if r < args.threshold]:
        poly, notes, ref = grow_field(ndvi, transform, x, y, t, args.radius // PIXEL_M)
        if poly is None:
            if best is None:
                best = (Point(x, y).buffer(15, cap_style="square"), notes, t)
            break  # a stricter threshold found nothing: keep the previous result
        best = (poly, notes, t)
        merged = any(n.startswith(("large", "irregular")) for n in notes)
        if not merged or np.nanmax(ref) < 0.3:  # villages stay ragged; don't shrink them
            break
    return best


def calibrate(known_path, ndvi, transform, crs, args):
    known = gpd.read_file(known_path).to_crs(crs)
    thresholds = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    print(f"\nCalibration on {len(known)} known fields (IoU = overlap / union; 1.0 = identical):")
    print(f"  {'threshold':>9} " + " ".join(f"{n[-12:]:>12}" for n in known["name"]) + "   mean IoU")
    for t in thresholds:
        ious = []
        for geom in known.geometry:
            seed = geom.representative_point()
            poly, _, _ = grow_field(ndvi, transform, seed.x, seed.y, t, args.radius // PIXEL_M)
            ious.append(0.0 if poly is None else poly.intersection(geom).area / poly.union(geom).area)
        print(f"  {t:9.2f} " + " ".join(f"{v:12.2f}" for v in ious) + f"   {np.mean(ious):8.2f}")


def false_colour(it, arrays, clear):
    """NIR-red-green false colour: healthy crops bright red, bare soil brown-grey, water dark."""
    img = np.dstack([reflectance(arrays[b], it) for b in ("B08", "B04", "B03")])
    img = np.clip(img / np.array([0.5, 0.25, 0.25]), 0, 1)
    img[~clear] = 0.85  # clouds and shadows as light grey
    return img


def review_pages(fields, raw, transform, args):
    """Each field outlined on false colour in March and in September, 15 fields per page."""
    def clear_share(entry):
        return entry[2].mean()
    march = max([e for e in raw if e[0].datetime.month in (2, 3)], key=clear_share, default=raw[0])
    autumn = max([e for e in raw if e[0].datetime.month in (9, 10)], key=clear_share, default=raw[-1])
    images = [(march[0].datetime.date(), false_colour(*march)), (autumn[0].datetime.date(), false_colour(*autumn))]

    half = 45  # pixels either side of the point: a 900 m chip
    per_page, cols = 15, 3
    pages = []
    for start in range(0, len(fields), per_page):
        chunk = fields.iloc[start:start + per_page]
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.3, rows * 2.6), squeeze=False)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        for k, (_, f) in enumerate(chunk.iterrows()):
            col, row = (int(v) for v in ~transform * (f.seed_x, f.seed_y))
            for j, (day, img) in enumerate(images):
                ax = axes[k // cols, (k % cols) * 2 + j]
                ax.axis("on")
                ax.imshow(img[max(row - half, 0):row + half, max(col - half, 0):col + half])
                off_c, off_r = max(col - half, 0), max(row - half, 0)
                if f.geometry is not None and not f.geometry.is_empty:
                    xs, ys = f.geometry.exterior.xy
                    c, r = ~transform * (np.array(xs), np.array(ys))
                    ax.plot(c - off_c, r - off_r, color="yellow", lw=1.3)
                ax.plot(col - off_c, row - off_r, "o", ms=3.5, mfc="cyan", mec="black", mew=0.6)
                if j == 0:
                    flag = "" if f.auto_quality == "ok" else " ⚠"
                    ax.set_title(f"{f['name']}{' TEST' if f.role == 'test' else ''}  {f.auto_acres:.1f} ac{flag}",
                                 fontsize=7.5, loc="left", color="crimson" if flag else "black")
                ax.set_xlabel(f"{day:%d %b %Y}", fontsize=6.5)
        fig.suptitle("Auto-drawn fields (yellow) around each sample point (cyan). False colour: crops red, "
                     "soil grey-brown, water dark. ⚠ = check the note in fields_auto.geojson", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(args.out, f"review_{start // per_page + 1:02d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        pages.append(path)
    return pages


def main():
    args = parse_args()
    points = gpd.read_file(args.points)
    crs = points.estimate_utm_crs()
    utm = points.to_crs(crs)
    area = box(*utm.total_bounds).buffer(args.radius + 100, join_style="mitre")
    dates, ndvi, transform, raw = monthly_stack(area, crs, args)
    clear_share = (~np.isnan(ndvi)).mean(axis=(1, 2))
    print("  months: " + ", ".join(f"{d:%b %y} {c:.0%}" for d, c in zip(dates, clear_share)) + " clear")

    if args.calibrate:
        calibrate(args.calibrate, ndvi, transform, crs, args)
        return

    polys, qualities, acres, used = [], [], [], []
    for p in utm.geometry:
        poly, notes, t = grow_with_retry(ndvi, transform, p.x, p.y, args)
        if poly.area / 4047 < MIN_ACRES and not any(n.startswith("never green") for n in notes):
            # A point on a ridge, path or field edge mixes two fields in its reference profile, so
            # almost nothing matches it. A point on a boundary belongs to the field it touches: try
            # starting points up to 20 m away and keep the largest clean field that touches the point.
            options = []
            for dx in (-20, -10, 0, 10, 20):
                for dy in (-20, -10, 0, 10, 20):
                    if dx or dy:
                        q, qn, qt = grow_with_retry(ndvi, transform, p.x + dx, p.y + dy, args)
                        clean = not any(n.startswith(("large", "irregular", "no field")) for n in qn)
                        if clean and q.distance(p) <= PIXEL_M:
                            options.append((q.area, q, qn, qt, math.hypot(dx, dy)))
            if options:
                area, poly, notes, t, moved = max(options, key=lambda o: o[0])
                notes = notes + [f"point on a field edge: grown from {moved:.0f} m away"]
        polys.append(poly)
        qualities.append("; ".join(notes) or "ok")
        acres.append(poly.area / 4047)
        used.append(t)
    fields = utm.copy()
    fields["seed_x"], fields["seed_y"] = utm.geometry.x, utm.geometry.y
    fields["auto_acres"], fields["auto_quality"], fields["auto_threshold"] = acres, qualities, used
    fields = fields.set_geometry(polys)

    os.makedirs(args.out, exist_ok=True)
    out = fields.to_crs("EPSG:4326")
    features = []
    for _, f in out.iterrows():
        props = {k: f[k] for k in ("name", "block", "role", "sample")}
        props.update(auto_acres=round(float(f.auto_acres), 2), auto_quality=f.auto_quality,
                     auto_threshold=float(f.auto_threshold))
        props.update({k: "" for k in LABEL_KEYS})
        ring = [[round(x, 7), round(y, 7)] for x, y in f.geometry.exterior.coords]
        features.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    path = os.path.join(args.out, "fields_auto.geojson")
    with open(path, "w") as fh:
        fh.write('{\n  "type": "FeatureCollection",\n  "features": [\n')
        fh.write(",\n".join("    " + json.dumps(ft) for ft in features))
        fh.write("\n  ]\n}\n")

    pages = review_pages(fields, raw, transform, args)
    counts = fields.auto_quality.str.split("; ").explode().value_counts()
    retried = sum(t < args.threshold for t in used)
    print(f"\n{len(fields)} fields grown at threshold {args.threshold} ({retried} retried stricter): "
          f"median {np.median(acres):.1f} acres")
    print("  quality: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"Saved {path} and {len(pages)} review pages in ./{args.out}/")


if __name__ == "__main__":
    main()
