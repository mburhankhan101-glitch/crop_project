"""
Step 11a: Sentinel-1 radar time series for the 100 labelled fields, next to their optical series.

Radar sends its own microwaves and measures how much bounces back (backscatter). It sees through
cloud, fog and darkness, so it has no gaps in winter fog or the monsoon. Two polarisations:
    VV  sent and received vertically: sensitive to soil moisture and surface roughness
    VH  sent vertically, received horizontally: mostly from volume scattering in leaves and stems,
        so it rises as a crop grows
    RVI radar vegetation index, 4 VH / (VV + VH), roughly 0 for bare soil and higher for dense canopy

Data: Sentinel-1 RTC (terrain-corrected gamma0) from Planetary Computer, one viewing geometry only
(--orbit), because backscatter depends on the angle the radar looks from. Field values are means in
linear power over the field's pixels (averaging tames speckle), converted to dB afterwards.

    python s1_fields.py

Outputs in output/radar/: fields_s1.csv, radar_vs_ndvi.png, radar_example.png, radar_cv.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping
from sklearn.model_selection import LeaveOneGroupOut

from s2_classify import MERGE, cross_predict, hand_features, load, models, scores
from s2_cube import lonlat_polygon, square
from s2_indices import STAC_URL

CLASSES = ["wheat", "other_crop", "not_cropped"]
COLOURS = {"wheat": "#c9962b", "other_crop": "#2c774c", "not_cropped": "#7a7a7a"}
OPTICAL_GAPS = [("2025-12-08", "2026-01-21"), ("2026-07-16", "2026-08-14")]  # shared by almost every field
MONTHS = [(2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--known", default="fields.geojson")
    p.add_argument("--size", type=int, default=10_000)
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--end", default="2026-09-30")
    p.add_argument("--orbit", default="descending:34", help="orbit state and relative orbit to use")
    p.add_argument("--buffer", type=float, default=5, help="shrink field outlines by this many metres")
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--matrix", default=os.path.join("output", "labels", "series_matrix.csv"))
    p.add_argument("--obs", default=os.path.join("output", "labels", "fields.csv"))
    p.add_argument("--cache", default=os.path.join("cache", "radar"))
    p.add_argument("--out", default=os.path.join("output", "radar"))
    return p.parse_args()


def radar_items(geometry, start, end, orbit):
    state, rel = orbit.split(":")
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=["sentinel-1-rtc"], intersects=geometry, datetime=f"{start}/{end}").item_collection()
    keep = [i for i in items if i.properties["sat:orbit_state"] == state and i.properties["sat:relative_orbit"] == int(rel)]
    return sorted(keep, key=lambda i: i.datetime)


def load_square(items, crs, bounds, args):
    """VV and VH (linear gamma0) for the whole square on the Sentinel-2 10 m grid, cached."""
    ids = np.array([i.id for i in items])
    path = os.path.join(args.cache, f"square_{args.orbit.replace(':', '_')}_{args.start}_{args.end}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["ids"], ids):
            return z["vv"], z["vh"], pd.to_datetime(z["dates"])
    x0, y0, x1, y1 = bounds
    cube = odc.stac.load(items, bands=["vv", "vh"], crs=str(crs), resolution=10, x=(x0, x1), y=(y0, y1),
                         groupby="solar_day", resampling="nearest", chunks={"time": 1, "x": 1000, "y": 1000},
                         fail_on_error=False).compute(num_workers=16)
    vv, vh = cube["vv"].values.astype("float32"), cube["vh"].values.astype("float32")
    dates = pd.to_datetime(cube.time.values).normalize()
    os.makedirs(args.cache, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ids=ids, vv=vv, vh=vh, dates=dates.values.astype("datetime64[D]"))
    os.replace(tmp, path)
    return vv, vh, dates


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(x)


def field_series(vv, vh, dates, fields, bounds, buffer):
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    rows = []
    for _, f in fields.iterrows():
        inside = ~geometry_mask([mapping(f.geometry.buffer(-buffer))], out_shape=vv.shape[1:], transform=transform)
        if inside.sum() == 0:
            continue
        for t, d in enumerate(dates):
            a, b = vv[t][inside], vh[t][inside]
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            if ok.sum() < max(3, 0.8 * inside.sum()):
                continue
            mv, mh = a[ok].mean(), b[ok].mean()
            rows.append({"field": f["name"], "date": d.date(), "pixels": int(ok.sum()), "VV_db": round(float(db(mv)), 3),
                         "VH_db": round(float(db(mh)), 3), "ratio_db": round(float(db(mh / mv)), 3),
                         "RVI": round(float(4 * mh / (mv + mh)), 4)})
    return pd.DataFrame(rows)


def shade_gaps(ax):
    for a, b in OPTICAL_GAPS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d9d9", alpha=0.6, lw=0)


def plot_classes(radar, matrix, labels, path):
    """Median NDVI (5-day, optical) and median VH and RVI (radar) per class."""
    cols = [c for c in matrix.columns if c.startswith("NDVI_")]
    dates = pd.to_datetime([c[5:] for c in cols])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for cls in CLASSES:
        names = labels.index[labels == cls]
        axes[0].plot(dates, matrix.loc[matrix.index.intersection(names), cols].median(), color=COLOURS[cls], lw=2,
                     label=f"{cls.replace('_', ' ')} (n={len(names)})")
        r = radar[radar["field"].isin(names)].groupby("date")[["VH_db", "RVI"]].median()
        axes[1].plot(pd.to_datetime(r.index), r["VH_db"], color=COLOURS[cls], lw=2, marker=".")
        axes[2].plot(pd.to_datetime(r.index), r["RVI"], color=COLOURS[cls], lw=2, marker=".")
    for ax, label in zip(axes, ("NDVI (optical, 5-day series)", "VH backscatter (dB)", "RVI (radar)")):
        shade_gaps(ax)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].set_title("Class medians: optical vs radar. Grey: periods with no usable optical images (winter fog, monsoon)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example(radar, obs, name, label, path):
    o = obs[(obs["field"] == name) & (obs["flag"] == "ok")]
    r = radar[radar["field"] == name]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    shade_gaps(ax)
    ax.plot(pd.to_datetime(o["date"]), o["NDVI_median"], "o-", color="#2c774c", ms=3, label="NDVI (optical, clear dates)")
    ax.set_ylabel("NDVI")
    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(r["date"]), r["VH_db"], "s-", color="#7b3294", ms=3, label="VH (radar)")
    ax2.set_ylabel("VH backscatter (dB)")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left")
    ax.set_title(f"{name} ({label}): radar keeps measuring through the gaps where optical has nothing")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def radar_features(radar, names):
    """Monthly means (October to May) of VH, VV and RVI per field: 24 features."""
    r = radar.copy()
    r["month"] = pd.to_datetime(r["date"]).dt.to_period("M")
    out = pd.DataFrame(index=names)
    for y, m in MONTHS:
        mon = r[r["month"] == pd.Period(f"{y}-{m:02d}")].groupby("field")[["VH_db", "VV_db", "RVI"]].mean()
        for c in mon.columns:
            out[f"{c}_{y}_{m:02d}"] = mon[c]
    return out


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    crs, bounds = square(args)
    items = radar_items(lonlat_polygon(crs, bounds), args.start, args.end, args.orbit)
    vv, vh, dates = load_square(items, crs, bounds, args)
    missing = [str(d.date()) for t, d in enumerate(dates) if not (np.isfinite(vv[t]).any() and np.isfinite(vh[t]).any())]
    print(f"{len(dates)} radar dates ({args.orbit}), {dates[0].date()} to {dates[-1].date()}"
          + (f"; failed to read (skipped): {', '.join(missing)}" if missing else ""))

    fields = gpd.read_file(args.fields).to_crs(crs)
    radar = field_series(vv, vh, dates, fields, bounds, args.buffer)
    radar.to_csv(os.path.join(args.out, "fields_s1.csv"), index=False)
    gap_dates = [d for d in dates if any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in OPTICAL_GAPS)]
    print(f"{radar['field'].nunique()} fields, {len(radar)} field-dates; {len(gap_dates)} radar dates fall in the "
          f"optical gaps ({', '.join(str(d.date()) for d in gap_dates)})")

    # How radar relates to NDVI: pairs of a clear optical observation and a radar date within 3 days.
    obs = pd.read_csv(args.obs, parse_dates=["date"])
    ok = obs[obs["flag"] == "ok"][["field", "date", "NDVI_median"]]
    ok = ok.assign(date=ok["date"].astype("datetime64[ns]")).sort_values("date")
    rr = radar.assign(date=pd.to_datetime(radar["date"]).astype("datetime64[ns]")).sort_values("date")
    pairs = pd.merge_asof(rr, ok, on="date", by="field", tolerance=pd.Timedelta(days=3), direction="nearest").dropna()
    print(f"\n{len(pairs)} radar/optical pairs within 3 days. Correlation with NDVI:")
    for c in ("VH_db", "VV_db", "ratio_db", "RVI"):
        print(f"  {c:<9} r = {pairs[c].corr(pairs['NDVI_median']):+.2f}")
    pairs.to_csv(os.path.join(args.out, "radar_ndvi_pairs.csv"), index=False)

    # Labels (merged Rabi classes) for the plots and the classification experiment.
    cargs = argparse.Namespace(fields=args.fields, matrix=args.matrix, season_start="2025-10-01",
                               season_end="2026-05-31", max_missing=0.5, binary=False)
    matrix, y_all, everyone = load(cargs)
    plot_classes(radar, matrix, y_all, os.path.join(args.out, "radar_vs_ndvi.png"))
    example = y_all.index[(y_all == "wheat") & (everyone.loc[y_all.index, "role"] == "train")][0]
    plot_example(radar, obs, example, "wheat", os.path.join(args.out, "radar_example.png"))

    # Can radar alone tell the classes apart? Spatial CV on the training fields only (the test set is spent).
    props = everyone.loc[y_all.index]
    is_train = (props["role"] == "train").values
    y, groups = y_all[is_train], props.loc[is_train, "block"]
    optical = hand_features(matrix, cargs).loc[y_all.index]
    rad = radar_features(radar, y_all.index)
    gap = rad[[c for c in rad.columns if c.endswith(("_2025_12", "_2026_01"))]]
    sets = {"optical (16)": optical, "radar (24)": rad, "radar, Dec-Jan only (6)": gap,
            "optical + radar (40)": optical.join(rad)}
    splits = list(LeaveOneGroupOut().split(optical[is_train], y, groups))
    rows = []
    print("\nSpatial CV on the training fields (balanced accuracy; labels came from optical curves):")
    for set_name, X in sets.items():
        for model in ("logistic", "random_forest"):
            pred = cross_predict(models()[model], X[is_train], y, groups, splits)
            acc, bal = scores(y, pred)
            rows.append({"features": set_name, "model": model, "accuracy": round(acc, 3), "balanced": round(bal, 3),
                         **{f"recall_{c}": round(float(np.mean(pred[y == c] == c)), 3) for c in CLASSES}})
            print(f"  {set_name:<26} {model:<14} acc {acc:.0%}  bal {bal:.0%}  "
                  + "  ".join(f"{c} {np.mean(pred[y == c] == c):.0%}" for c in CLASSES))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "radar_cv.csv"), index=False)
    print(f"\nSaved fields_s1.csv, radar_ndvi_pairs.csv, radar_vs_ndvi.png, radar_example.png, radar_cv.csv in {args.out}")


if __name__ == "__main__":
    main()
