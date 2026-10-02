"""
Step 8: classify each field's Rabi 2026 crop from its 5-day series, and measure it honestly.

    X  one row per field: output/labels/series_matrix.csv (set A) or hand-made features (set B)
    y  labels/fields.geojson rabi_2026, merged into wheat / other_crop / not_cropped
       (unknown is left out; --binary gives wheat / not_wheat)

Every model is scored by leave-one-strip-out CV on the training fields (spatial, the number we
choose by) and by repeated random 5-fold CV (for comparison only: the gap is the neighbour effect).
Settings such as C are tuned inside the training strips, never on the strip being predicted.

The winner is fixed before the test set is opened: the simplest model within one standard error
of the best spatial balanced accuracy (fewer features first, then logistic < SVM < forest < boosting).
Only then:

    python s2_classify.py            # CV on training fields only
    python s2_classify.py --test     # the one look at the blind test fields

--test refuses to run a second time (the test set would stop being blind) unless --again.
"""
import argparse
import json
import os
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

INDEX_NAMES = ["NDVI", "EVI", "NDMI", "NDRE", "NDWI"]
MERGE = {
    "wheat": "wheat",
    "other_winter": "other_crop", "short_winter": "other_crop", "sugarcane": "other_crop", "orchard": "other_crop",
    "non_crop": "not_cropped", "fallow": "not_cropped",
}
SEED = 0
RANDOM_REPEATS = 5
MONTHS = [(2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--matrix", default=os.path.join("output", "labels", "series_matrix.csv"))
    p.add_argument("--season-start", default="2025-10-01", help="first date used as a feature")
    p.add_argument("--season-end", default="2026-05-31", help="last date used as a feature")
    p.add_argument("--max-missing", type=float, default=0.5, help="drop dates missing for more than this share")
    p.add_argument("--binary", action="store_true", help="wheat vs not_wheat instead of three classes")
    p.add_argument("--test", action="store_true", help="evaluate the chosen model on the blind test fields")
    p.add_argument("--again", action="store_true", help="allow --test a second time (the test set is no longer blind)")
    p.add_argument("--out", default=os.path.join("output", "classify"))
    return p.parse_args()


# ---------- data ----------

def load(args):
    fc = json.load(open(args.fields))
    props = pd.DataFrame([f["properties"] for f in fc["features"]]).set_index("name")
    matrix = pd.read_csv(args.matrix, index_col=0)
    props = props.loc[matrix.index.intersection(props.index)]
    labelled = props[props["rabi_2026"].isin(MERGE)]  # unknown and unlabelled fields are left out
    y = labelled["rabi_2026"].map(MERGE)
    if args.binary:
        y = y.where(y == "wheat", "not_wheat")
    return matrix, y, props


def column_date(col):
    return date.fromisoformat(col.split("_", 1)[1])


def curve_features(matrix, args):
    """Set A: every index on every 5-day date of the season (dates missing for most fields dropped)."""
    start, end = date.fromisoformat(args.season_start), date.fromisoformat(args.season_end)
    cols = [c for c in matrix.columns if start <= column_date(c) <= end]
    cols = [c for c in cols if matrix[c].isna().mean() <= args.max_missing]
    return matrix[cols]


def hand_features(matrix, args):
    """Set B: ~16 numbers a crop scientist would look at, computed from the same series."""
    def series(index):
        cols = [c for c in matrix.columns if c.startswith(index + "_")]
        s = matrix[cols]
        s.columns = [column_date(c) for c in cols]
        return s

    def between(s, a, b):
        return s[[d for d in s.columns if a <= d <= b]]

    def month(s, y, m):
        return s[[d for d in s.columns if (d.year, d.month) == (y, m)]].mean(axis=1)

    ndvi, evi, ndmi, ndre, ndwi = (series(i) for i in INDEX_NAMES)
    season0 = date.fromisoformat(args.season_start)
    out = pd.DataFrame(index=matrix.index)
    for y, m in MONTHS:
        out[f"ndvi_{date(y, m, 1):%b}".lower()] = month(ndvi, y, m)
    core = between(ndvi, date(2025, 12, 1), date(2026, 4, 30))
    out["ndvi_peak"] = core.max(axis=1)
    peak_day = core.fillna(-9).values.argmax(axis=1)
    out["peak_day"] = [(core.columns[i] - season0).days for i in peak_day]
    out.loc[core.isna().all(axis=1), "peak_day"] = np.nan
    out["days_above_05"] = (between(ndvi, date(2025, 11, 1), date(2026, 4, 30)) > 0.5).sum(axis=1) * 5
    out["ndvi_min_sowing"] = between(ndvi, date(2025, 10, 15), date(2025, 12, 5)).min(axis=1)
    out["ndmi_mar"] = month(ndmi, 2026, 3)
    out["evi_peak"] = between(evi, date(2025, 12, 1), date(2026, 4, 30)).max(axis=1)
    out["ndre_feb"] = month(ndre, 2026, 2)
    out["ndwi_mar"] = month(ndwi, 2026, 3)
    return out


# ---------- models ----------

def models():
    """name -> (pipeline, grid of settings tuned inside the training strips, uses sample weights)."""
    impute = lambda: SimpleImputer(strategy="median")
    return {
        "baseline": (make_pipeline(impute(), DummyClassifier(strategy="most_frequent")), {}, False),
        "logistic": (make_pipeline(impute(), StandardScaler(),
                                   LogisticRegression(class_weight="balanced", max_iter=5000)),
                     {"logisticregression__C": [0.01, 0.1, 1, 10]}, False),
        "svm_rbf": (make_pipeline(impute(), StandardScaler(),
                                  SVC(class_weight="balanced", probability=False, random_state=SEED)),
                    {"svc__C": [0.1, 1, 10, 100]}, False),
        "random_forest": (make_pipeline(impute(), RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)), {}, False),
        "grad_boost": (make_pipeline(impute(), GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)), {}, True),
    }


def fit(spec, X, y, groups):
    """Fit one model; if it has settings, pick them by leave-one-strip-out inside these fields only."""
    pipe, grid, weighted = spec
    fit_params = {}
    if weighted:
        fit_params[pipe.steps[-1][0] + "__sample_weight"] = compute_sample_weight("balanced", y)
    if not grid:
        return clone(pipe).fit(X, y, **fit_params)
    search = GridSearchCV(clone(pipe), grid, cv=LeaveOneGroupOut(), scoring="balanced_accuracy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an inner fold can miss a rare class entirely
        search.fit(X, y, groups=groups, **fit_params)
    return search.best_estimator_


def cross_predict(spec, X, y, groups, splits, keep=None):
    """Out-of-fold predictions: each field is predicted by a model that never saw it.
    keep (a mask) limits which fields are used for training; all held-out fields are predicted."""
    pred = pd.Series(index=y.index, dtype=object)
    for train, held in splits:
        if keep is not None:
            train = train[keep[train]]
        model = fit(spec, X.iloc[train], y.iloc[train], groups.iloc[train])
        pred.iloc[held] = model.predict(X.iloc[held])
    return pred


def choose(results, n, sizes):
    """One-standard-error rule: of the models within one standard error of the top spatial score,
    take the simplest (fewer features first, then the simpler model). Differences smaller than
    the noise are not evidence, and simpler models generalise better."""
    contenders = results[results["model"] != "baseline"].copy()
    top = contenders["spatial_bal"].max()
    se = np.sqrt(top * (1 - top) / n)
    contenders["size"] = contenders["features"].map(sizes)
    contenders["order"] = contenders["model"].map(list(models()).index)
    near = contenders[contenders["spatial_bal"] >= top - se]
    return near.sort_values(["size", "order"]).iloc[0], top, se, near


def scores(y, pred):
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred)


def margin(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


# ---------- plots ----------

def plot_confusions(panels, classes, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2))
    for ax, (title, y, pred) in zip(np.atleast_1d(axes), panels):
        cm = confusion_matrix(y, pred, labels=classes)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks(range(len(classes)), classes, rotation=20)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("label")
        acc, bal = scores(y, pred)
        ax.set_title(f"{title}\naccuracy {acc:.0%}, balanced {bal:.0%}, n={len(y)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model, X, y, feature_set, title, path):
    clf = model.steps[-1][1]
    if hasattr(clf, "feature_importances_"):
        imp, how = clf.feature_importances_, "impurity importance"
    elif hasattr(clf, "coef_"):
        imp, how = np.abs(clf.coef_).mean(axis=0), "mean |coefficient| (standardised features)"
    else:
        r = permutation_importance(model, X, y, scoring="balanced_accuracy", n_repeats=20, random_state=SEED)
        imp, how = r.importances_mean, "permutation importance (training fields)"
    imp = pd.Series(imp, index=X.columns)
    if feature_set == "A":
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for index in INDEX_NAMES:
            part = imp[[c for c in imp.index if c.startswith(index + "_")]]
            ax.plot([column_date(c) for c in part.index], part.values, marker=".", label=index)
        ax.legend(ncol=5, fontsize=9)
        ax.set_ylabel(how)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.sort_values().plot.barh(ax=ax, color="#4c72b0")
        ax.set_xlabel(how)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(matrix, y, pred, labels_from, path):
    """NDVI curves of misclassified fields over the median curve of each class."""
    cols = [c for c in matrix.columns if c.startswith("NDVI_")]
    dates = [column_date(c) for c in cols]
    wrong = y.index[y != pred]
    classes = sorted(y.unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 4), sharey=True)
    for ax, cls in zip(np.atleast_1d(axes), classes):
        ax.plot(dates, matrix.loc[y.index[y == cls], cols].median(), color="black", lw=3, label=f"median {cls}")
        for name in wrong[y[wrong] == cls]:
            ax.plot(dates, matrix.loc[name, cols], marker=".", ms=3, label=f"{name} -> {pred[name]}")
        ax.set_title(f"labelled {cls} ({labels_from}): {sum(y[wrong] == cls)} wrong", fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("NDVI")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- main ----------

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    matrix, y_all, everyone = load(args)
    props = everyone.loc[y_all.index]
    full = {"A": curve_features(matrix, args), "B": hand_features(matrix, args)}  # all fields, unknown too
    sets = {k: v.loc[props.index] for k, v in full.items()}
    is_train = (props["role"] == "train").values
    y, groups = y_all[is_train], props.loc[is_train, "block"]
    conf = props.loc[is_train, "rabi_2026_confidence"]
    classes = sorted(y.unique())
    print(f"Training fields: {len(y)} (" + ", ".join(f"{c} {sum(y == c)}" for c in classes) + ")")
    print(f"Feature set A: {sets['A'].shape[1]} columns (5 indices x {sets['A'].shape[1] // 5} dates), "
          f"set B: {sets['B'].shape[1]} hand-made features")

    spatial = list(LeaveOneGroupOut().split(sets["A"][is_train], y, groups))
    rows, preds = [], {}
    for fs, X_all in sets.items():
        X = X_all[is_train]
        for name, spec in models().items():
            pred = cross_predict(spec, X, y, groups, spatial)
            preds[(fs, name)] = pred
            acc, bal = scores(y, pred)
            per_strip = [accuracy_score(y.iloc[h], pred.iloc[h]) for _, h in spatial]
            rnd = []
            for r in range(RANDOM_REPEATS):
                folds = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
                rnd.append(scores(y, cross_predict(spec, X, y, groups, list(folds.split(X, y)))))
            racc, rbal = np.mean(rnd, axis=0)
            recall = {c: np.mean(pred[y == c] == c) for c in classes}
            rows.append({"features": fs, "model": name, "spatial_acc": acc, "spatial_bal": bal,
                         "random_acc": racc, "random_bal": rbal, "gap_bal": rbal - bal,
                         "strip_min": min(per_strip), "strip_max": max(per_strip),
                         **{f"recall_{c}": v for c, v in recall.items()}})
            print(f"  {fs} {name:<14} spatial acc {acc:.0%} bal {bal:.0%} | random acc {racc:.0%} bal {rbal:.0%}"
                  f" | strips {min(per_strip):.0%}-{max(per_strip):.0%}")
    results = pd.DataFrame(rows)
    results.round(3).to_csv(os.path.join(args.out, "cv_results.csv"), index=False)

    n = len(y)
    best, top, se, near = choose(results, n, {f: X.shape[1] for f, X in sets.items()})
    fs, name = best["features"], best["model"]
    print(f"\nTop spatial balanced accuracy {top:.0%}, standard error {se:.1%}: "
          f"{len(near)} models within one SE; the simplest is chosen")
    base = results[(results["model"] == "baseline")].iloc[0]
    print(f"Baseline (always {y.mode()[0]}): accuracy {base['spatial_acc']:.0%}, balanced {base['spatial_bal']:.0%}")
    print(f"Chosen by spatial CV: set {fs} + {name}: accuracy {best['spatial_acc']:.0%} "
          f"(+/- {margin(best['spatial_acc'], n):.0%}), balanced {best['spatial_bal']:.0%}")
    print(f"  random CV says balanced {best['random_bal']:.0%}: the neighbour effect is {best['gap_bal']:+.0%}")
    print("  recall: " + ", ".join(f"{c} {best[f'recall_{c}']:.0%}" for c in classes))

    # Label confidence: train on high + medium labels only (scored on the same held-out fields).
    spec = models()[name]
    X = sets[fs][is_train]
    clean = (conf != "low").values
    pred_clean = cross_predict(spec, X, y, groups, spatial, keep=clean)
    pred_all = preds[(fs, name)]
    hm = conf != "low"
    print(f"\nLabel confidence ({clean.sum()} high/medium of {n} labels):")
    rows = []
    for label, p in (("trained on all labels", pred_all), ("trained on high/medium", pred_clean)):
        a, b = scores(y, p)
        a2, b2 = scores(y[hm], p[hm])
        rows.append({"training": label, "n_train": n if p is pred_all else int(clean.sum()),
                     "acc_all": a, "bal_all": b, "acc_highmed": a2, "bal_highmed": b2})
        print(f"  {label:<24} all fields: acc {a:.0%} bal {b:.0%} | high/medium fields: acc {a2:.0%} bal {b2:.0%}")
    pd.DataFrame(rows).round(3).to_csv(os.path.join(args.out, "label_confidence.csv"), index=False)

    cv = pd.DataFrame({"true": y, "pred": pred_all, "block": groups, "confidence": conf,
                       "label": props.loc[is_train, "rabi_2026"]})
    cv.to_csv(os.path.join(args.out, "cv_predictions.csv"), index_label="field")
    wrong = cv[cv["true"] != cv["pred"]]
    print(f"\nSpatial-CV mistakes: {len(wrong)} of {n}")
    print("  " + wrong.drop(columns="block").to_string().replace("\n", "\n  "))

    final = fit(spec, X, y, groups)
    tuned = {k: v for k, v in final.get_params().items() if k in spec[1]}
    print(f"\nFinal model trained on all {n} training fields" + (f", tuned {tuned}" if tuned else ""))
    plot_importance(final, X, y, fs, f"What the final model uses: set {fs} + {name}",
                    os.path.join(args.out, "importance.png"))
    plot_errors(matrix, y, pred_all, "curve labels, spatial CV", os.path.join(args.out, "errors_cv.png"))
    panels = [("Spatial CV (training fields)", y, pred_all)]

    if args.test:
        out = os.path.join(args.out, "test_predictions.csv")
        if os.path.exists(out) and not args.again:
            raise SystemExit(f"{out} exists: the test set has been looked at once already. "
                             "Use --again only if you accept that it is no longer blind.")
        is_test = props["role"] == "test"
        yt = y_all[is_test]
        pt = pd.Series(final.predict(sets[fs][is_test]), index=yt.index)
        tc = props.loc[is_test, "rabi_2026_confidence"]
        a, b = scores(yt, pt)
        print(f"\nBLIND TEST ({len(yt)} fields, labels from true-colour only):")
        print(f"  accuracy {a:.0%} (+/- {margin(a, len(yt)):.0%}, so {a - margin(a, len(yt)):.0%}"
              f"-{min(1, a + margin(a, len(yt))):.0%}), balanced {b:.0%}")
        print(f"  majority baseline on the test fields: {np.mean(yt == y.mode()[0]):.0%}")
        print("  recall: " + ", ".join(f"{c} {np.mean(pt[yt == c] == c):.0%} ({sum(yt == c)})" for c in classes))
        for level in ("high", "medium", "low"):
            m = tc == level
            if m.any():
                print(f"  {level}-confidence labels: {np.mean(yt[m] == pt[m]):.0%} of {m.sum()}")
        unknown = everyone[(everyone["role"] == "test") & (everyone["rabi_2026"] == "unknown")].index
        if len(unknown):
            pu = final.predict(full[fs].loc[unknown])
            print("  unknown test fields, for interest: " + ", ".join(f"{u} -> {p}" for u, p in zip(unknown, pu)))
        pd.DataFrame({"true": yt, "pred": pt, "confidence": tc, "label": props.loc[is_test, "rabi_2026"],
                      "notes": props.loc[is_test, "notes"].str[:80]}).to_csv(out, index_label="field")
        panels.append(("Blind test (once)", yt, pt))
        plot_errors(matrix, yt, pt, "blind labels", os.path.join(args.out, "errors_test.png"))
        print(f"  saved {out}")

    else:
        saved = os.path.join(args.out, "test_predictions.csv")
        if os.path.exists(saved):  # redraw the earlier test result; no new predictions are made
            t = pd.read_csv(saved, index_col=0)
            panels.append(("Blind test (saved result)", t["true"], t["pred"]))

    plot_confusions(panels, classes, os.path.join(args.out, "confusion.png"))
    print(f"\nSaved cv_results.csv, cv_predictions.csv, confusion.png, importance.png, errors_*.png in {args.out}")


if __name__ == "__main__":
    main()
