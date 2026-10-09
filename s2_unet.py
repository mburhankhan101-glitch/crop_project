"""
Step 13: a U-Net that maps whole image patches, trained from 68 labelled fields.

Input: the 16 set-B features of every pixel (from the Step 10 cube, cache/cube/features.npy) as 16
image channels. A U-Net looks at a patch at once: its encoder shrinks the patch to see wider context
(is this pixel inside a big field, next to a road?), its decoder grows it back to full resolution,
and skip connections carry the fine detail across, so every pixel gets a class.

Labels are sparse: only pixels inside the labelled training fields have a class; the loss ignores
the rest. Two variants are compared:
    sparse   the labelled field pixels only
    pseudo   plus pseudo-labels: pixels where a per-pixel logistic regression, retrained in each fold
             on that fold's training fields only, is at least --pseudo-conf sure (weighted --pseudo-weight)

Evaluation is the usual leave-one-strip-out CV: training crops come only from the training strips,
one strip is held back for early stopping, and the held-out strip is mapped in one piece. Scores are
per labelled pixel and per field (majority vote), next to the per-pixel logistic regression.
Finished folds are cached, so a run that stops early (--budget) continues where it left off.

    python s2_unet.py

Outputs in output/unet/: cv_results.csv, crop_map_unet.tif, unet_vs_pixels.png
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
import rasterio
import torch
from matplotlib.colors import ListedColormap
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch import nn

import s2_cube
from s2_classify import MERGE

CLASSES = ["not_cropped", "other_crop", "wheat"]
COLOURS = {"wheat": "#c9962b", "other_crop": "#2c774c", "not_cropped": "#9a9a9a", "uncertain": "#e8dfd0"}
STRIP_PX = 200   # the 2 km cross-validation strips are 200 pixels wide
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", type=int, default=64, help="training patch size in pixels")
    p.add_argument("--crops-per-epoch", type=int, default=192)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pseudo-conf", type=float, default=0.95)
    p.add_argument("--pseudo-weight", type=float, default=0.3)
    p.add_argument("--min-prob", type=float, default=0.6, help="as in Step 10: below this a pixel is 'uncertain'")
    p.add_argument("--budget", type=float, default=8.5, help="minutes; stop starting new fits after this")
    p.add_argument("--cache", default=os.path.join("cache", "unet"))
    p.add_argument("--out", default=os.path.join("output", "unet"))
    return p.parse_args()


# ---------- data ----------

def load_inputs():
    """16 feature channels (standardised, gaps filled with the channel median), labels, strips and geometry."""
    cargs = s2_cube.parse_args([])
    crs, bounds = s2_cube.square(cargs)
    X = np.load(os.path.join(cargs.cache, "features.npy")).astype("float32")
    med = np.nanmedian(X, axis=(1, 2), keepdims=True)
    X = np.where(np.isfinite(X), X, med)
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    n = X.shape[1]
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    fields = gpd.read_file(cargs.fields).to_crs(crs)
    labels = np.full((n, n), -1, dtype="int64")      # -1 = no label, ignored by the loss
    field_id = np.full((n, n), -1, dtype="int32")
    rows = []
    for k, (_, f) in enumerate(fields.iterrows()):
        inside = ~geometry_mask([mapping(f.geometry.buffer(-5))], out_shape=(n, n), transform=transform)
        field_id[inside] = k
        cls = MERGE.get(f["rabi_2026"])
        rows.append({"k": k, "field": f["name"], "role": f["role"], "block": f["block"], "label": cls})
        if f["role"] == "train" and cls:
            labels[inside] = CLASSES.index(cls)
    meta = pd.DataFrame(rows).set_index("k")
    strip_of_col = np.array([f"strip_{c // STRIP_PX + 1}" for c in range(n)])
    return X, labels, field_id, meta, strip_of_col, crs, bounds, cargs


def column_runs(cols):
    """Contiguous [start, stop) runs of True in a boolean column mask."""
    runs, start = [], None
    for i, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def random_crops(region_cols, n_rows, size, count, rng, must_have, labels):
    """Patch corners fully inside the allowed columns, each containing at least one labelled pixel."""
    runs = [(a, b) for a, b in column_runs(region_cols) if b - a >= size]
    out = []
    tries = 0
    while len(out) < count and tries < count * 50:
        tries += 1
        a, b = runs[rng.integers(len(runs))]
        c = rng.integers(a, b - size + 1)
        r = rng.integers(0, n_rows - size + 1)
        if must_have[r:r + size, c:c + size].any():
            out.append((r, c))
    return out


# ---------- the U-Net ----------

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    """Two downsamplings: each pixel's decision can use about 40 pixels (400 m) of context."""

    def __init__(self, cin, n_out=3, base=16, dropout=0.2):
        super().__init__()
        self.e1, self.e2, self.e3 = block(cin, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), block(4 * base, 2 * base)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * base, base, 2, 2), block(2 * base, base)
        self.out = nn.Conv2d(base, n_out, 1)

    def forward(self, x):
        a = self.e1(x)                      # full resolution
        b = self.e2(self.pool(a))           # half
        c = self.drop(self.e3(self.pool(b)))  # quarter: the widest view
        d = self.d2(torch.cat([self.u2(c), b], 1))   # back up, with the skip connection from b
        return self.out(self.d1(torch.cat([self.u1(d), a], 1)))


def predict_area(model, X, cols):
    """Class probabilities for whole columns of the square, padded to a multiple of 4."""
    part = X[:, :, cols[0]:cols[1]]
    h, w = part.shape[1:]
    ph, pw = (-h) % 4, (-w) % 4
    part = np.pad(part, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(part[None])), 1)[0].numpy()
    return p[:, :h, :w]


def train_unet(X, target, weight, train_cols, val_lab, args, class_w):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = UNet(X.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = torch.tensor(class_w, dtype=torch.float32)
    has_target = target >= 0
    vcols = column_span(val_lab >= 0, X.shape[2])
    best, best_state, wait, curve = np.inf, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        corners = random_crops(train_cols, X.shape[1], args.crop, args.crops_per_epoch, rng, has_target, None)
        total = 0.0
        for i in range(0, len(corners), args.batch):
            batch = corners[i:i + args.batch]
            xb = torch.from_numpy(np.stack([X[:, r:r + args.crop, c:c + args.crop] for r, c in batch]))
            yb = torch.from_numpy(np.stack([target[r:r + args.crop, c:c + args.crop] for r, c in batch]))
            wb = torch.from_numpy(np.stack([weight[r:r + args.crop, c:c + args.crop] for r, c in batch]))
            logits = model(xb)
            per_px = nn.functional.cross_entropy(logits, yb.clamp(min=0), weight=cw, reduction="none")
            loss = (per_px * wb * (yb >= 0)).sum() / ((wb * (yb >= 0)).sum() + 1e-6)   # unlabelled pixels: weight 0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        p = predict_area(model, X, vcols)
        lab = val_lab[:, vcols[0]:vcols[1]]
        m = lab >= 0
        val = float(np.mean(-np.log(np.take_along_axis(p, np.where(m, lab, 0)[None], 0)[0][m] + 1e-8)
                            * class_w[lab[m]]) / np.mean(class_w[lab[m]]))
        curve.append((total / max(1, len(corners) // args.batch), val))
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


def strip_fields(meta, strip):
    """Indices of the labelled training fields whose sample point lies in this strip."""
    return meta.index[(meta.role == "train") & meta.label.notna() & (meta.block == strip)].to_numpy()


def labels_of(labels, field_id, ks):
    """The label raster restricted to these fields (all their pixels, whichever strip they fall in)."""
    out = np.full_like(labels, -1)
    m = np.isin(field_id, ks)
    out[m] = labels[m]
    return out


def column_span(mask, n, margin=8):
    cols = np.flatnonzero(mask.any(axis=0))
    return max(cols[0] - margin, 0), min(cols[-1] + 1 + margin, n)


def field_votes(pred, cols, field_id, meta, ks):
    """Majority class of each of these fields, from a prediction covering columns cols."""
    sub = field_id[:, cols[0]:cols[1]]
    return {meta.loc[k, "field"]: CLASSES[np.bincount(pred[sub == k], minlength=3).argmax()] for k in ks}


# ---------- main ----------

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    X, labels, field_id, meta, strip_of_col, crs, bounds, cargs = load_inputs()
    n = X.shape[1]
    flat = X.reshape(X.shape[0], -1).T
    strips = sorted(set(strip_of_col))
    print(f"16 channels x {n} x {n}; labelled pixels {int((labels >= 0).sum())} in "
          f"{meta[(meta.role == 'train') & meta.label.notna()].shape[0]} training fields")

    stopped = False
    for k, held in enumerate(strips):
        val_strip = strips[(k + 1) % len(strips)]
        held_cols, val_cols = strip_of_col == held, strip_of_col == val_strip
        train_cols = ~held_cols & ~val_cols          # training crops come only from these columns
        train_ks = np.concatenate([strip_fields(meta, s) for s in strips if s not in (held, val_strip)])
        held_ks, val_ks = strip_fields(meta, held), strip_fields(meta, val_strip)
        cols = column_span(np.isin(field_id, held_ks), n)
        path = os.path.join(args.cache, f"fold_{held}.npz")
        if os.path.exists(path):
            continue
        if (time.time() - t0) / 60 > args.budget:
            stopped = True
            break
        # The per-pixel logistic regression, trained on this fold's training fields only (also the teacher).
        train_lab = labels_of(labels, field_id, train_ks)
        m = train_lab.ravel() >= 0
        lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(flat[m], train_lab.ravel()[m])
        proba = lr.predict_proba(flat).T.reshape(3, n, n)
        class_w = len(train_lab[train_lab >= 0]) / (3 * np.bincount(train_lab[train_lab >= 0], minlength=3))
        preds = {"logistic": proba[:, :, cols[0]:cols[1]].argmax(0)}
        curves = {}
        for variant in ("sparse", "pseudo"):
            target, weight = train_lab.copy(), (train_lab >= 0).astype("float32")
            if variant == "pseudo":
                conf = (proba.max(0) >= args.pseudo_conf) & (train_lab < 0) & (field_id < 0)
                conf[:, ~train_cols] = False
                target[conf] = proba.argmax(0)[conf]
                weight[conf] = args.pseudo_weight
            model, curve = train_unet(X, target, weight, train_cols, labels_of(labels, field_id, val_ks), args, class_w)
            preds[variant] = predict_area(model, X, cols).argmax(0)
            curves[variant] = np.array(curve)
        np.savez_compressed(path, cols=np.array(cols), **{f"pred_{v}": p for v, p in preds.items()},
                            **{f"curve_{v}": c for v, c in curves.items()})
        print(f"  fold {held}: epochs sparse {len(curves['sparse'])}, pseudo {len(curves['pseudo'])} "
              f"({time.time() - t0:.0f} s)", flush=True)
    if stopped:
        print(f"Time budget reached; run again to continue ({sum(os.path.exists(os.path.join(args.cache, f'fold_{s}.npz')) for s in strips)} of {len(strips)} folds done)")
        return

    # Scores over all folds: labelled pixels and fields of the held-out strips.
    rows, fields_out = [], []
    for model_name in ("logistic", "sparse", "pseudo"):
        y_px, p_px, y_f, p_f = [], [], [], []
        for held in strips:
            z = np.load(os.path.join(args.cache, f"fold_{held}.npz"))
            cols = tuple(z["cols"])
            pred = z[f"pred_{model_name}"]
            ks = strip_fields(meta, held)
            lab = labels_of(labels, field_id, ks)[:, cols[0]:cols[1]]
            y_px.append(lab[lab >= 0])
            p_px.append(pred[lab >= 0])
            for f, v in field_votes(pred, cols, field_id, meta, ks).items():
                truth = meta.loc[meta.field == f, "label"].iloc[0]
                y_f.append(truth)
                p_f.append(v)
                fields_out.append({"model": model_name, "field": f, "strip": held, "true": truth, "pred": v})
        y_px, p_px = np.concatenate(y_px), np.concatenate(p_px)
        rows.append({"model": model_name, "pixel_balanced": round(balanced_accuracy_score(y_px, p_px), 3),
                     "field_accuracy": round(float(np.mean(np.array(y_f) == np.array(p_f))), 3),
                     "field_balanced": round(balanced_accuracy_score(y_f, p_f), 3),
                     "fields_wrong": ", ".join(f for f, a, b in zip([r["field"] for r in fields_out if r["model"] == model_name], y_f, p_f) if a != b)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "cv_results.csv"), index=False)
    pd.DataFrame(fields_out).to_csv(os.path.join(args.out, "field_predictions.csv"), index=False)
    print("\nSpatial CV (leave one strip out), labelled training fields:")
    print(res.to_string(index=False))

    # The final map: U-Net (pseudo-labels) trained on four strips with the fifth watching for overfitting,
    # then the whole square; compared with the per-pixel logistic map.
    all_lab = labels.copy()
    m = all_lab.ravel() >= 0
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(flat[m], all_lab.ravel()[m])
    proba = lr.predict_proba(flat).T.reshape(3, n, n)
    class_w = len(all_lab[m.reshape(n, n)]) / (3 * np.bincount(all_lab[all_lab >= 0], minlength=3))
    val_cols = strip_of_col == strips[-1]
    train_cols = ~val_cols
    target = labels_of(labels, field_id, np.concatenate([strip_fields(meta, s) for s in strips[:-1]]))
    weight = (target >= 0).astype("float32")
    conf = (proba.max(0) >= args.pseudo_conf) & (field_id < 0)
    conf[:, ~train_cols] = False
    target[conf] = proba.argmax(0)[conf]
    weight[conf] = args.pseudo_weight
    model, curve = train_unet(X, target, weight, train_cols, labels_of(labels, field_id, strip_fields(meta, strips[-1])),
                              args, class_w)
    p_unet = predict_area(model, X, (0, n))
    maps = {}
    for name, p in (("pixel logistic", proba), ("U-Net", p_unet)):
        cls = p.argmax(0).astype("uint8") + 1
        cls[p.max(0) < args.min_prob] = 4
        maps[name] = cls
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    code = {1: "not_cropped", 2: "other_crop", 3: "wheat", 4: "uncertain"}
    cmap_rgb = {k: tuple(int(COLOURS[v][i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k, v in code.items()}
    with rasterio.open(os.path.join(args.out, "crop_map_unet.tif"), "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=0, compress="deflate") as dst:
        dst.write(maps["U-Net"], 1)
        dst.write_colormap(1, cmap_rgb)

    def speckle(cls):
        """Share of pixels that disagree with the majority of their 3 x 3 neighbourhood."""
        counts = np.stack([np.pad(cls == c, 1).astype("int8") for c in range(1, 5)])
        win = sum(counts[:, i:i + n, j:j + n] for i in range(3) for j in range(3))
        return float(np.mean(win.argmax(0) + 1 != cls))

    print("\nWhole-square maps (no 3 x 3 smoothing on either):")
    stats = []
    for name, cls in maps.items():
        share = {code[c]: float(np.mean(cls == c)) for c in code}
        stats.append({"map": name, **{k: round(v, 3) for k, v in share.items()}, "speckle": round(speckle(cls), 3)})
        print(f"  {name:<15} wheat {share['wheat']:.1%}  other {share['other_crop']:.1%}  not cropped "
              f"{share['not_cropped']:.1%}  uncertain {share['uncertain']:.1%}  speckle {speckle(cls):.1%}")
    pd.DataFrame(stats).to_csv(os.path.join(args.out, "map_stats.csv"), index=False)

    # Figure: a 2.5 km window around the original fields, both maps side by side.
    known = gpd.read_file(cargs.known).to_crs(crs)
    c = known.union_all().centroid
    size = 250
    col0 = int((c.x - bounds[0]) // 10) - size // 2
    row0 = int((bounds[3] - c.y) // 10) - size // 2
    cmap = ListedColormap([COLOURS[code[k]] for k in (1, 2, 3, 4)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (name, cls) in zip(axes, maps.items()):
        ax.imshow(cls[row0:row0 + size, col0:col0 + size], cmap=cmap, vmin=0.5, vmax=4.5, interpolation="nearest")
        sp = next(s["speckle"] for s in stats if s["map"] == name)
        un = next(s["uncertain"] for s in stats if s["map"] == name)
        ax.set_title(f"{name}: uncertain {un:.0%}, speckle {sp:.0%} (whole square)")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[code[k]], label=code[k].replace("_", " ")) for k in (3, 2, 1, 4)]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False)
    fig.suptitle("Same 2.5 km window, same 16 features: each pixel on its own vs a U-Net that sees its neighbourhood")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "unet_vs_pixels.png"), dpi=140)
    plt.close(fig)
    print(f"Saved cv_results.csv, field_predictions.csv, map_stats.csv, crop_map_unet.tif, unet_vs_pixels.png "
          f"in {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
