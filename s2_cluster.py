"""
Group the training fields by the shape of their NDVI year, to speed up labelling them.

k-means on each field's 5-day NDVI series (from s2_series.py). Fields in a cluster share a
crop calendar, so a cluster can be named once from its average curve and the rules in
LABELS.md, and then only the fields far from their cluster's centre need a closer look.

Test fields are left out entirely: if their labels helped name the clusters, they would leak
into the training labels (see LABELS.md, blind test protocol).

Dates missing for any field (the monsoon gap, and before the first observation) are dropped,
so every field is compared on the same dates.

    python s2_cluster.py                    # try k = 3..10, use the best silhouette
    python s2_cluster.py --k 8

Writes to ./output/labels/:
    clusters.csv   one row per training field: cluster, distance to the cluster centre
    clusters.png   every cluster's member curves and average, and the k comparison
"""
import argparse
import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from s2_indices import shade_seasons


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default=os.path.join("output", "labels", "series_matrix.csv"))
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--index", default="NDVI", help="which index's series to cluster (default NDVI)")
    p.add_argument("--k", type=int, help="number of clusters (default: best silhouette for k = 3..10)")
    p.add_argument("--max-missing", type=float, default=0.2,
                   help="leave out fields missing more than this share of dates (default 0.2)")
    p.add_argument("--out", default=os.path.join("output", "labels"))
    return p.parse_args()


def main():
    args = parse_args()
    matrix = pd.read_csv(args.matrix, index_col="field")
    fields = gpd.read_file(args.fields).set_index("name")
    train = fields.index[fields["role"] == "train"]

    cols = [c for c in matrix.columns if c.startswith(f"{args.index}_")]
    X = matrix.loc[matrix.index.intersection(train), cols]
    # Dates missing for most fields (the winter fog and monsoon gaps) say nothing about one field.
    X = X.loc[:, X.isna().mean() <= 0.5]
    missing = X.isna().mean(axis=1)
    left_out = list(missing[missing > args.max_missing].index)
    X = X[missing <= args.max_missing]
    X = X.loc[:, X.notna().all()]
    dates = pd.to_datetime([c.split("_", 1)[1] for c in X.columns])
    print(f"{len(X)} training fields x {X.shape[1]} dates of {args.index} "
          f"({len(cols) - X.shape[1]} dates dropped: missing for some field)")
    if left_out:
        print(f"  left out, too many missing dates: {', '.join(left_out)}")

    scores = {}
    for k in range(3, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
        scores[k] = (km.inertia_, silhouette_score(X, km.labels_))
    print("  k  inertia  silhouette   (silhouette: -1..1, higher = clusters better separated)")
    for k, (inertia, sil) in scores.items():
        print(f"  {k:2d} {inertia:8.1f}  {sil:9.3f}")
    k = args.k or max(scores, key=lambda k: scores[k][1])
    print(f"  -> using k = {k}" + (" (set with --k)" if args.k else " (best silhouette)"))

    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
    # Number clusters by when they peak, so the order means something across runs.
    order = np.argsort([dates[np.argmax(c)] for c in km.cluster_centers_])
    renumber = {old: new for new, old in enumerate(order)}
    cluster = np.array([renumber[c] for c in km.labels_])
    centres = km.cluster_centers_[order]
    distance = np.linalg.norm(X.to_numpy() - centres[cluster], axis=1) / np.sqrt(X.shape[1])  # RMS, NDVI units

    out = pd.DataFrame({"field": X.index, "cluster": cluster, "distance": distance.round(4)})
    out["rank_in_cluster"] = out.groupby("cluster")["distance"].rank().astype(int)
    for col in ("rabi_2026", "kharif_2026", "auto_acres"):
        out[col] = fields.loc[out["field"], col].to_numpy()
    out = out.sort_values(["cluster", "distance"])
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "clusters.csv"), index=False)

    ncols = 3
    nrows = int(np.ceil((k + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.1 * nrows), squeeze=False)
    for c in range(k):
        ax = axes.ravel()[c]
        shade_seasons(ax, dates[0].date(), dates[-1].date())
        members = out[out["cluster"] == c]
        for name in members["field"]:
            ax.plot(dates, X.loc[name], color="grey", lw=0.6, alpha=0.6)
        ax.plot(dates, centres[c], color="black", lw=2.2)
        ax.set_xlim(dates[0], dates[-1])  # the season shading would otherwise stretch the axis
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(f"cluster {c}: {len(members)} fields, spread {members['distance'].mean():.2f}", fontsize=9,
                     loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10, 12, 2, 4, 6, 8]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    ax = axes.ravel()[k]
    ks = list(scores)
    ax.plot(ks, [scores[j][1] for j in ks], "-o", color="black", ms=4)
    ax.axvline(k, color="grey", ls=":")
    ax.set_title("silhouette by k (higher is better)", fontsize=9, loc="left")
    ax.set_xlabel("k", fontsize=8)
    ax.tick_params(labelsize=7)
    for ax in axes.ravel()[k + 1:]:
        ax.axis("off")
    fig.suptitle(f"k-means on the training fields' {args.index} years (test fields left out). Grey = fields, "
                 "black = cluster average; shading = Rabi / Kharif. Gaps in the axis are dropped dates.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out, "clusters.png"), dpi=120)
    plt.close(fig)

    print("\nClusters (fields listed from the centre outwards):")
    for c in range(k):
        m = out[out["cluster"] == c]
        peak = dates[np.argmax(centres[c])]
        print(f"  {c}: {len(m):2d} fields, peak {centres[c].max():.2f} in {peak:%b %Y}, "
              f"low {centres[c].min():.2f}; " + ", ".join(m["field"]))
    print(f"Saved clusters.csv and clusters.png in ./{args.out}/")


if __name__ == "__main__":
    main()
