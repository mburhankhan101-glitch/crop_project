"""
Step 12: first neural networks with PyTorch, on pixels from the labelled training fields.

Three models, all judged by the same leave-one-strip-out spatial CV as Step 8:
    logistic   logistic regression on each pixel's 16 set-B features (the baseline)
    mlp        a small neural network on the same 16 features: 16 -> 64 -> 64 -> 3
    tempcnn    a 1-D convolutional network on the raw 5-day curves (5 indices x 49 dates,
               October to May), which learns its own features instead of using ours

Pixels come from the Step 10 cube (cached blocks), inside each field outline shrunk by 5 m, at most
--per-field per field so that large villages do not outweigh small wheat fields. Pixels of one field
are near-copies, so every split is by strip, and scores are reported per pixel and per field
(majority vote of the field's pixels). The networks stop early when the loss on a validation strip
(one of the training strips) stops improving. The blind test fields are not used: that set is spent.

    python s2_torch.py

Outputs in output/torch/: cv_results.csv, field_predictions.csv, learning_curves.png
"""
import argparse
import os
import time

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

import s2_cube
from s2_classify import load

CLASSES = ["not_cropped", "other_crop", "wheat"]
INDEX_NAMES = ["NDVI", "EVI", "NDMI", "NDRE", "NDWI"]
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-field", type=int, default=200, help="at most this many pixels per field")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8, help="stop after this many epochs without validation improvement")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularisation on the weights")
    p.add_argument("--series-start", default="2025-10-01")
    p.add_argument("--series-end", default="2026-05-31")
    p.add_argument("--cache", default=os.path.join("cache", "torch"))
    p.add_argument("--out", default=os.path.join("output", "torch"))
    return p.parse_args()


# ---------- pixels from the cube ----------

def extract_pixels(args):
    """Features and 5-day series for every pixel inside the labelled fields, from the cached cube blocks."""
    path = os.path.join(args.cache, "pixels.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["field"], z["features"], z["series"], list(z["feature_names"]), list(z["dates"])
    cargs = s2_cube.parse_args([])
    crs, bounds = s2_cube.square(cargs)
    items = s2_cube.search_geometry(s2_cube.lonlat_polygon(crs, bounds), cargs.start, cargs.end, cargs.max_cloud)
    obs = pd.read_csv(cargs.obs, parse_dates=["date"])
    hazy = sorted(obs.loc[obs["flag"].str.contains("haze"), "date"].dt.date.unique())
    hazy_days = np.array([(d - s2_cube.date(2025, 1, 1)).days for d in hazy])
    grid = pd.date_range(cargs.grid_start, cargs.grid_end, freq="5D")
    model_args = argparse.Namespace(fields=cargs.fields, matrix=cargs.matrix, season_start="2025-10-01",
                                    season_end="2026-05-31", max_missing=0.5, binary=False)
    _, _, _, feature_names = s2_cube.final_model(cargs)
    keep_dates = [d for d in grid if pd.Timestamp(args.series_start) <= d <= pd.Timestamp(args.series_end)]
    series_cols = [f"{n}_{d.date()}" for n in INDEX_NAMES for d in keep_dates]

    n = cargs.size // 10
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    fields = gpd.read_file(cargs.fields).to_crs(crs)
    owner = np.full((n, n), -1, dtype="int32")  # which field each pixel belongs to
    for k, (_, f) in enumerate(fields.iterrows()):
        inside = ~geometry_mask([mapping(f.geometry.buffer(-5))], out_shape=(n, n), transform=transform)
        owner[inside] = k
    names = fields["name"].to_numpy()
    halo = cargs.cloud_buffer
    out_field, out_feat, out_series = [], [], []
    t0 = time.time()
    for r in range(0, n, cargs.block):
        for c in range(0, n, cargs.block):
            rows, cols = slice(r, min(r + cargs.block, n)), slice(c, min(c + cargs.block, n))
            sub = owner[rows, cols]
            if (sub < 0).all():
                continue
            raw, days = s2_cube.load_block(items, crs, bounds, rows, cols, halo, cargs)
            feats, _, matrix = s2_cube.block_features(raw, days, hazy_days, grid, model_args, cargs, with_series=True)
            H, W = raw["SCL"].shape[1:]
            top, left = rows.start - max(rows.start - halo, 0), cols.start - max(cols.start - halo, 0)
            full = np.full((H, W), -1, dtype="int32")
            full[top:top + sub.shape[0], left:left + sub.shape[1]] = sub
            idx = np.flatnonzero(full.ravel() >= 0)
            out_field.append(names[full.ravel()[idx]])
            out_feat.append(feats.iloc[idx][feature_names].to_numpy("float32"))
            out_series.append(matrix.iloc[idx][series_cols].to_numpy("float32").reshape(len(idx), len(INDEX_NAMES), -1))
            print(f"  block rows {r}, cols {c}: {len(idx)} field pixels ({time.time() - t0:.0f} s)", flush=True)
    field, features, series = np.concatenate(out_field), np.concatenate(out_feat), np.concatenate(out_series)
    os.makedirs(args.cache, exist_ok=True)
    np.savez_compressed(path, field=field, features=features, series=series, feature_names=np.array(feature_names),
                        dates=np.array([str(d.date()) for d in keep_dates]))
    return field, features, series, feature_names, [str(d.date()) for d in keep_dates]


# ---------- models ----------

class MLP(nn.Module):
    def __init__(self, n_in, n_out=3, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class TempCNN(nn.Module):
    """Three 1-D convolutions sliding along time (each sees 5 dates = 25 days at once), then a dense layer."""

    def __init__(self, channels, length, n_out=3, filters=32, kernel=5, dense=128, dropout=0.3):
        super().__init__()
        layers, c = [], channels
        for _ in range(3):
            layers += [nn.Conv1d(c, filters, kernel, padding=kernel // 2), nn.BatchNorm1d(filters), nn.ReLU(),
                       nn.Dropout(dropout)]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * length, dense), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(dense, n_out))

    def forward(self, x):
        return self.head(self.conv(x))


def standardise(train, *others):
    """Scale with the training pixels' mean and spread (per feature or per index channel); gaps become 0 = average."""
    axis = (0, 2) if train.ndim == 3 else 0
    mean = np.nanmean(train, axis=axis, keepdims=True)
    std = np.nanstd(train, axis=axis, keepdims=True) + 1e-6
    return [np.nan_to_num((a - mean) / std, nan=0.0).astype("float32") for a in (train, *others)]


def train_net(make, X_tr, y_tr, X_va, y_va, args, weights):
    """Mini-batch Adam with weighted cross-entropy; keep the weights from the epoch with the lowest validation loss."""
    torch.manual_seed(SEED)
    model = make()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)
    gen = torch.Generator().manual_seed(SEED)
    best, best_state, wait, curve = np.inf, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), args.batch):
            b = order[i:i + args.batch]
            loss = loss_fn(model(Xt[b]), yt[b])   # forward + loss
            opt.zero_grad()                       # clear old gradients
            loss.backward()                       # backpropagation
            opt.step()                            # move every weight downhill
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        curve.append((total / len(Xt), val))
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, curve


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).argmax(dim=1).numpy()


def field_vote(fields, pred):
    """Majority class of each field's pixels."""
    df = pd.DataFrame({"field": fields, "pred": pred})
    return df.groupby("field")["pred"].agg(lambda p: np.bincount(p, minlength=3).argmax())


# ---------- main ----------

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    field, features, series, feature_names, dates = extract_pixels(args)

    cargs = argparse.Namespace(fields=os.path.join("labels", "fields.geojson"),
                               matrix=os.path.join("output", "labels", "series_matrix.csv"),
                               season_start="2025-10-01", season_end="2026-05-31", max_missing=0.5, binary=False)
    _, y_all, everyone = load(cargs)
    train_fields = y_all.index[everyone.loc[y_all.index, "role"] == "train"]
    rng = np.random.default_rng(SEED)
    keep = []
    for f in train_fields:  # cap pixels per field
        idx = np.flatnonzero(field == f)
        keep.append(rng.choice(idx, size=min(len(idx), args.per_field), replace=False) if len(idx) else idx)
    keep = np.sort(np.concatenate(keep))
    field, features, series = field[keep], features[keep], series[keep]
    y = np.array([CLASSES.index(y_all[f]) for f in field])
    strip = everyone.loc[field, "block"].to_numpy()
    per_field = pd.Series(field).value_counts()
    print(f"{len(field)} pixels from {len(per_field)} training fields (median {per_field.median():.0f}, "
          f"max {per_field.max()} per field); series {series.shape[1]} indices x {series.shape[2]} dates")

    strips = sorted(set(strip))
    names = ("logistic", "mlp", "tempcnn")
    pixel_pred = {m: np.empty(len(y), dtype=int) for m in names}   # every pixel predicted once, when held out
    field_rows, curves = [], {}
    t0 = time.time()
    for k, held in enumerate(strips):
        val_strip = strips[(k + 1) % len(strips)]          # one training strip watches for overfitting
        te, va = strip == held, strip == val_strip
        tr = ~te & ~va
        weights = compute_class_weight("balanced", classes=np.arange(3), y=y[tr])
        F_tr, F_va, F_te = standardise(features[tr], features[va], features[te])
        S_tr, S_va, S_te = standardise(series[tr], series[va], series[te])
        lr = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0)
        lr.fit(np.vstack([F_tr, F_va]), np.concatenate([y[tr], y[va]]))  # no early stopping, so it can use both
        pixel_pred["logistic"][te] = lr.predict(F_te)
        mlp, c1 = train_net(lambda: MLP(F_tr.shape[1]), F_tr, y[tr], F_va, y[va], args, weights)
        pixel_pred["mlp"][te] = predict(mlp, F_te)
        cnn, c2 = train_net(lambda: TempCNN(S_tr.shape[1], S_tr.shape[2]), S_tr, y[tr], S_va, y[va], args, weights)
        pixel_pred["tempcnn"][te] = predict(cnn, S_te)
        curves[held] = {"mlp": c1, "tempcnn": c2}
        for name in names:
            p = pixel_pred[name][te]
            for f, v in field_vote(field[te], p).items():
                field_rows.append({"field": f, "strip": held, "model": name, "true": y_all[f], "pred": CLASSES[v],
                                   "pixel_agreement": round(float(np.mean(p[field[te] == f] == v)), 2)})
        print(f"  held-out {held} (validation {val_strip}): {te.sum()} pixels; epochs mlp {len(c1)}, tempcnn {len(c2)} "
              f"({time.time() - t0:.0f} s)", flush=True)

    # Overall pixel and field scores.
    summary = []
    fr = pd.DataFrame(field_rows)
    for name in names:
        g = fr[fr["model"] == name]
        summary.append({"model": name, "pixel_accuracy": round(float(np.mean(pixel_pred[name] == y)), 3),
                        "pixel_balanced": round(balanced_accuracy_score(y, pixel_pred[name]), 3),
                        "field_accuracy": round(float(np.mean(g["true"] == g["pred"])), 3),
                        "field_balanced": round(balanced_accuracy_score(g["true"], g["pred"]), 3),
                        "fields_wrong": ", ".join(g.loc[g["true"] != g["pred"], "field"])})
    res = pd.DataFrame(summary)
    res.to_csv(os.path.join(args.out, "cv_results.csv"), index=False)
    fr.to_csv(os.path.join(args.out, "field_predictions.csv"), index=False)
    print("\nSpatial CV (leave one strip out), training fields only:")
    print(res.to_string(index=False))

    # Learning curves of the first held-out strip.
    held = strips[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name in zip(axes, ("mlp", "tempcnn")):
        c = np.array(curves[held][name])
        ax.plot(range(1, len(c) + 1), c[:, 0], label="training loss")
        ax.plot(range(1, len(c) + 1), c[:, 1], label="validation loss (another strip)")
        best = int(np.argmin(c[:, 1])) + 1
        ax.axvline(best, color="grey", ls="--", lw=1)
        ax.text(best, ax.get_ylim()[1], f" kept epoch {best}", va="top", fontsize=8, color="grey")
        ax.set_title(f"{name.upper()}, fold holding out {held}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted cross-entropy")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curves.png"), dpi=130)
    plt.close(fig)
    print(f"Saved cv_results.csv, field_predictions.csv, learning_curves.png in {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
