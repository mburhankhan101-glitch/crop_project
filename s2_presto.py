"""
Step 14: a geospatial foundation model (Presto) against the hand-made baseline, with few labels.

Presto (NASA Harvest) is a small transformer (404,160 encoder weights) pretrained without labels on
millions of pixel time series of Sentinel-1, Sentinel-2, weather and elevation. Frozen, it turns a
field's monthly series into a 128-number embedding; a logistic regression on those embeddings is the
"linear probe". The question: does pretraining help when labels are scarce?

Per field and month (October 2025 to May 2026, 8 timesteps):
    Sentinel-2  median DN of clear observations over the field (from the Step 10 cube cache):
                B2, B3, B4 (RGB), B8 (NIR 10 m), B8A (NIR 20 m), and NDVI
    Sentinel-1  mean VV and VH in dB (from Step 11, output/radar/fields_s1.csv)
    masked      red edge (B5-B7) and SWIR (B11-B12) because B6, B7 and B12 are not in the cache
                (Presto masks whole groups), weather and elevation (not downloaded), and Sentinel-2
                in months without a clear observation
Weights: the official checkpoint from github.com/nasaharvest/presto (cache/external/presto),
loaded with weights_only=True.

Two experiments, both leave-one-strip-out on the 68 training fields (the blind test set is spent):
    1. all labels: logistic regression (C tuned in the training strips) on the 16 set-B features,
       on Presto embeddings, on both, and on the raw monthly inputs Presto was given (the control:
       same information, no pretraining)
    2. few labels: 3, 6, 12 and 24 training fields per fold (at least one per class), 30 random draws

    python s2_presto.py

Outputs in output/presto/: cv_results.csv, few_labels.csv, few_labels.png, embeddings.csv
"""
import argparse
import os
import sys
import time
from datetime import date

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import s2_cube
from s2_classify import cross_predict, hand_features, load, models
from s2_fields import CLOUD_SCL, grow
from s2_indices import CLEAR_SCL

PRESTO_DIR = os.path.join("cache", "external", "presto")
MONTHS = [(2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]
S2_USED = {"B02": 2, "B03": 3, "B04": 4, "B08": 8, "B8A": 9}    # cube band -> Presto channel
GROUPS = {"S1": [0, 1], "S2_RGB": [2, 3, 4], "S2_Red_Edge": [5, 6, 7], "S2_NIR_10m": [8], "S2_NIR_20m": [9],
          "S2_SWIR": [10, 11], "ERA5": [12, 13], "SRTM": [14, 15], "NDVI": [16]}
ADD = np.array([25, 25] + [0] * 10 + [-272.15, 0, 0, 0, 0], dtype="float32")
DIV = np.array([25, 25] + [1e4] * 10 + [35, 0.03, 2000, 50, 1], dtype="float32")
CLASSES = ["not_cropped", "other_crop", "wheat"]
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=30, help="random draws per few-label size")
    p.add_argument("--sizes", default="3,6,12,24", help="training fields per fold in the few-label experiment")
    p.add_argument("--radar", default=os.path.join("output", "radar", "fields_s1.csv"))
    p.add_argument("--cache", default=os.path.join("cache", "presto"))
    p.add_argument("--out", default=os.path.join("output", "presto"))
    return p.parse_args()


# ---------- inputs ----------

def field_monthly_s2(cache_path):
    """Median raw DN per field, month and band, over the clear observations of the field's pixels."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    cargs = s2_cube.parse_args([])
    crs, bounds = s2_cube.square(cargs)
    items = s2_cube.search_geometry(s2_cube.lonlat_polygon(crs, bounds), cargs.start, cargs.end, cargs.max_cloud)
    obs = pd.read_csv(cargs.obs, parse_dates=["date"])
    hazy = {(d - date(2025, 1, 1)).days for d in obs.loc[obs["flag"].str.contains("haze"), "date"].dt.date}
    n = cargs.size // 10
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    fields = gpd.read_file(cargs.fields).to_crs(crs)
    owner = np.full((n, n), -1, dtype="int32")
    for k, (_, f) in enumerate(fields.iterrows()):
        owner[~geometry_mask([mapping(f.geometry.buffer(-5))], out_shape=(n, n), transform=transform)] = k
    names = fields["name"].to_numpy()
    halo = cargs.cloud_buffer
    pieces = []
    for r in range(0, n, cargs.block):
        for c in range(0, n, cargs.block):
            rows, cols = slice(r, min(r + cargs.block, n)), slice(c, min(c + cargs.block, n))
            sub = owner[rows, cols]
            if (sub < 0).all():
                continue
            raw, days = s2_cube.load_block(items, crs, bounds, rows, cols, halo, cargs)
            cloud = np.stack([grow(np.isin(s, CLOUD_SCL), cargs.cloud_buffer) for s in raw["SCL"]])
            clear = np.isin(raw["SCL"], CLEAR_SCL) & ~cloud & ~np.isin(days, list(hazy))[:, None, None]
            top, left = rows.start - max(rows.start - halo, 0), cols.start - max(cols.start - halo, 0)
            h, w = sub.shape
            inner = (slice(None), slice(top, top + h), slice(left, left + w))
            ok = clear[inner]
            month = np.array([(date(2025, 1, 1).toordinal() + int(d)) for d in days])
            month = np.array([date.fromordinal(o).year * 100 + date.fromordinal(o).month for o in month])
            for k in np.unique(sub[sub >= 0]):
                pix = sub == k
                for t in range(len(days)):
                    v = ok[t][pix]
                    if not v.any():
                        continue
                    row = {"field": names[k], "ym": int(month[t]), "n": int(v.sum())}
                    for b in S2_USED:
                        row[b] = raw[b][inner][t][pix][v].astype("float32")
                    pieces.append(row)
            print(f"  block rows {r}, cols {c}", flush=True)
    # Median over all clear pixel-observations of the month (pooled across dates).
    df = pd.DataFrame(pieces)
    out = []
    for (f, ym), g in df.groupby(["field", "ym"]):
        out.append({"field": f, "ym": ym, "obs": int(g["n"].sum()),
                    **{b: float(np.median(np.concatenate(g[b].to_list()))) for b in S2_USED}})
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    res.to_csv(cache_path, index=False)
    return res


def presto_inputs(names, s2, radar, latlons):
    """x, mask (1 = ignore), dynamic world (9 = missing) for each field: [fields, 8 months, 17 channels]."""
    T = len(MONTHS)
    x = np.zeros((len(names), T, 17), dtype="float32")
    mask = np.ones_like(x)
    r = radar.copy()
    r["ym"] = pd.to_datetime(r["date"]).dt.year * 100 + pd.to_datetime(r["date"]).dt.month
    rm = r.groupby(["field", "ym"])[["VV_db", "VH_db"]].mean()
    sm = s2.set_index(["field", "ym"])
    for i, f in enumerate(names):
        for t, (y, m) in enumerate(MONTHS):
            ym = y * 100 + m
            if (f, ym) in rm.index:
                x[i, t, 0], x[i, t, 1] = rm.loc[(f, ym), "VV_db"], rm.loc[(f, ym), "VH_db"]
                mask[i, t, GROUPS["S1"]] = 0
            if (f, ym) in sm.index:
                for b, ch in S2_USED.items():
                    x[i, t, ch] = sm.loc[(f, ym), b] - 1000          # remove the processing-baseline offset
                for g in ("S2_RGB", "S2_NIR_10m", "S2_NIR_20m", "NDVI"):
                    mask[i, t, GROUPS[g]] = 0
    xn = (x + ADD) / DIV
    red, nir = xn[:, :, 4], xn[:, :, 8]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn[:, :, 16] = np.where(red + nir > 0, (nir - red) / (red + nir), 0)
    raw = np.where(mask == 1, np.nan, xn)             # for the no-pretraining control
    xn = np.where(mask == 1, 0, xn).astype("float32")
    dw = np.full((len(names), T), 9, dtype="int64")
    return (torch.from_numpy(xn), torch.from_numpy(mask), torch.from_numpy(dw), torch.from_numpy(latlons.astype("float32")),
            raw)


def embed(x, mask, dw, latlons):
    sys.path.insert(0, PRESTO_DIR)
    from single_file_presto import Presto
    model = Presto.construct()
    model.load_state_dict(torch.load(os.path.join(PRESTO_DIR, "data", "default_model.pt"), map_location="cpu",
                                     weights_only=True))
    enc = model.encoder.eval()
    # One field at a time: Presto needs the same number of masked tokens for every item in a batch,
    # and fields differ in how many months were too cloudy.
    with torch.no_grad():
        return np.vstack([enc(x[i:i + 1], dynamic_world=dw[i:i + 1], latlons=latlons[i:i + 1], mask=mask[i:i + 1],
                              month=MONTHS[0][1] - 1, eval_task=True).numpy() for i in range(len(x))])


# ---------- experiments ----------

def few_label_curve(sets, y, groups, sizes, draws):
    """Field balanced accuracy on the held-out strip when only k training fields are labelled."""
    rng = np.random.default_rng(SEED)
    splits = list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    rows = []
    for k in sizes:
        for d in range(draws):
            preds = {name: pd.Series(index=y.index, dtype=object) for name in sets}
            for tr, te in splits:
                tr_y = y.iloc[tr]
                # at least one field per class, then the rest at random
                pick = [rng.choice(np.flatnonzero(tr_y.values == c)) for c in CLASSES if (tr_y.values == c).any()]
                rest = np.setdiff1d(np.arange(len(tr)), pick)
                pick = np.concatenate([pick, rng.choice(rest, size=max(0, min(k, len(tr)) - len(pick)), replace=False)])
                idx = tr[pick.astype(int)]
                for name, X in sets.items():
                    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(class_weight="balanced", max_iter=5000))
                    clf.fit(X.iloc[idx], y.iloc[idx])
                    preds[name].iloc[te] = clf.predict(X.iloc[te])
            for name, p in preds.items():
                rows.append({"fields_per_fold": k, "draw": d, "features": name,
                             "balanced": balanced_accuracy_score(y, p.astype(str))})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cargs = argparse.Namespace(fields=os.path.join("labels", "fields.geojson"),
                               matrix=os.path.join("output", "labels", "series_matrix.csv"),
                               season_start="2025-10-01", season_end="2026-05-31", max_missing=0.5, binary=False)
    matrix, y_all, everyone = load(cargs)
    names = list(y_all.index)
    s2 = field_monthly_s2(os.path.join(args.cache, "field_monthly_s2.csv"))
    radar = pd.read_csv(args.radar)
    pts = gpd.read_file(os.path.join("labels", "sample_points.geojson")).set_index("name").loc[names]
    latlons = np.c_[pts.geometry.y, pts.geometry.x]
    x, mask, dw, ll, raw = presto_inputs(names, s2, radar, latlons)
    used = [c for g in ("S1", "S2_RGB", "S2_NIR_10m", "S2_NIR_20m", "NDVI") for c in GROUPS[g]]
    R = pd.DataFrame(raw[:, :, used].reshape(len(names), -1), index=names,
                     columns=[f"m{t}_c{c}" for t in range(len(MONTHS)) for c in used])
    emb = embed(x, mask, dw, ll)
    E = pd.DataFrame(emb, index=names, columns=[f"presto_{i}" for i in range(emb.shape[1])])
    E.round(5).to_csv(os.path.join(args.out, "embeddings.csv"), index_label="field")
    s2_months = (1 - mask[:, :, 2].numpy()).sum(axis=1)
    print(f"{len(names)} labelled fields; Sentinel-2 months per field: median {np.median(s2_months):.0f} of 8; "
          f"embedding {emb.shape[1]} numbers ({time.time() - t0:.0f} s)")

    props = everyone.loc[names]
    is_train = (props["role"] == "train").values
    y, groups = y_all[is_train], props.loc[is_train, "block"]
    B = hand_features(matrix, cargs).loc[names]
    sets = {"hand-made (16)": B[is_train], "Presto (128)": E[is_train], "both (144)": B[is_train].join(E[is_train]),
            f"raw monthly ({R.shape[1]})": R[is_train]}
    splits = list(LeaveOneGroupOut().split(B[is_train], y, groups))
    rows = []
    print("\n1. All 68 training fields, spatial CV, logistic regression with C tuned inside the training strips:")
    for name, X in sets.items():
        pred = cross_predict(models()["logistic"], X, y, groups, splits)
        bal = balanced_accuracy_score(y, pred)
        acc = float(np.mean(pred == y))
        rows.append({"features": name, "accuracy": round(acc, 3), "balanced": round(bal, 3),
                     "wrong": ", ".join(y.index[pred != y])})
        print(f"  {name:<16} accuracy {acc:.0%}  balanced {bal:.0%}  wrong: {', '.join(y.index[pred != y]) or '-'}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "cv_results.csv"), index=False)

    sizes = [int(s) for s in args.sizes.split(",")]
    curve = few_label_curve({k: v for k, v in sets.items() if k != "both (144)"}, y, groups, sizes, args.draws)
    raw_name = [k for k in sets if k.startswith("raw")][0]
    curve.to_csv(os.path.join(args.out, "few_labels.csv"), index=False)
    summary = curve.groupby(["fields_per_fold", "features"])["balanced"].agg(["mean", "std"]).round(3)
    print(f"\n2. Few labels: balanced accuracy on the held-out strip, mean and spread over {args.draws} draws:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, colour in (("hand-made (16)", "#c9962b"), ("Presto (128)", "#356fa8"), (raw_name, "#7a7a7a")):
        s = curve[curve["features"] == name].groupby("fields_per_fold")["balanced"]
        m, sd = s.mean(), s.std()
        ax.plot(m.index, m.values, "o-", color=colour, label=name)
        ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=0.2)
    full = {r["features"]: r["balanced"] for r in rows}
    ax.axhline(full["hand-made (16)"], color="#c9962b", ls=":", lw=1)
    ax.axhline(full["Presto (128)"], color="#356fa8", ls=":", lw=1)
    ax.axhline(1 / 3, color="grey", ls="--", lw=1, label="chance (balanced)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("labelled training fields per fold")
    ax.set_ylabel("balanced accuracy (held-out strip)")
    ax.set_title("Few labels: hand-made features vs the Presto foundation model\n"
                 "grey: Presto's inputs without pretraining; dotted: all training fields", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "few_labels.png"), dpi=130)
    plt.close(fig)
    print(f"\nSaved cv_results.csv, few_labels.csv, few_labels.png, embeddings.csv in {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
