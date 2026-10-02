"""
Step 9: figures, tables and numbers for the report in paper/, all generated from the outputs of
Steps 7 and 8 so that no number in the text is ever typed by hand.

    paper/figures/*.pdf          vector figures for LaTeX
    paper/generated/numbers.tex  \\newcommand macros, e.g. \\TestAcc -> 89\\%
    paper/generated/tab_*.tex    tables, included with \\input

Run after s2_classify.py --test:

    python s2_paper.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from s2_classify import (MERGE, choose, column_date, curve_features, fit, hand_features, load, margin,
                         models)

CLASSES = ["wheat", "other_crop", "not_cropped"]
COLOURS = {"wheat": "#c9962b", "other_crop": "#2c774c", "not_cropped": "#7a7a7a", "unknown": "#c8c8c8"}
MODEL_NAMES = {"baseline": "Majority baseline", "logistic": "Logistic regression (L2)", "svm_rbf": "SVM (RBF kernel)",
               "random_forest": "Random forest", "grad_boost": "Gradient boosting"}
TEXT_WIDTH = 5.5  # inches: the report's text block, so figures are drawn at their printed size
SHORT_NAMES = {"baseline": "Majority baseline", "logistic": "Logistic reg.", "svm_rbf": "SVM (RBF)",
               "random_forest": "Random forest", "grad_boost": "Gradient boosting"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--points", default=os.path.join("labels", "sample_points.geojson"))
    p.add_argument("--known", default="fields.geojson", help="the 3 fields the study area is centred on")
    p.add_argument("--matrix", default=os.path.join("output", "labels", "series_matrix.csv"))
    p.add_argument("--fields-csv", default=os.path.join("output", "labels", "fields.csv"))
    p.add_argument("--classify", default=os.path.join("output", "classify"))
    p.add_argument("--out", default="paper")
    return p.parse_args()


def whole(x):
    """A share as a whole percentage, rounded half up (0.985 -> 99, as in the README)."""
    return int(np.floor(100 * x + 0.5 + 1e-9))


def pct(x):
    return f"{whole(x)}\\%"


# ---------- figures ----------

def fig_area(points, known, labels, path):
    """Sample points by Rabi class and role, with the five 2 km strips used as CV blocks."""
    crs = known.estimate_utm_crs()
    centre = known.to_crs(crs).union_all().centroid
    x0, y0 = centre.x - 5000, centre.y - 5000
    pts = points.to_crs(crs)
    pts["cls"] = pts["name"].map(labels).fillna("unknown")
    fig, ax = plt.subplots(figsize=(2.55, 2.55))
    ax.spines[["top", "right"]].set_visible(True)
    for k in range(6):
        ax.axvline(2 * k, color="#999999", lw=0.6, ls="--")
    for k in range(5):
        ax.text(2 * k + 1, 10.25, f"strip {k + 1}", ha="center", va="bottom", fontsize=6, color="#555555")
    for cls in ["wheat", "other_crop", "not_cropped", "unknown"]:
        for role, marker in (("train", "o"), ("test", "^")):
            sel = pts[(pts["cls"] == cls) & (pts["role"] == role)]
            ax.scatter((sel.geometry.x - x0) / 1000, (sel.geometry.y - y0) / 1000, s=16, marker=marker,
                       color=COLOURS[cls], edgecolor="black", linewidth=0.3)
    handles = [plt.Line2D([], [], ls="", marker="o", color=COLOURS[c], markeredgecolor="black", markeredgewidth=0.3,
                          label=c.replace("_", " ")) for c in ["wheat", "other_crop", "not_cropped", "unknown"]]
    handles += [plt.Line2D([], [], ls="", marker=m, color="white", markeredgecolor="black", label=r)
                for m, r in (("o", "training"), ("^", "blind test"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
              columnspacing=0.8, handletextpad=0.2)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 10.2)
    ax.set_aspect("equal")
    ax.set_xlabel("km east (UTM 43N)")
    ax.set_ylabel("km north")
    fig.savefig(path)
    plt.close(fig)


def fig_profiles(matrix, y, path):
    """Median and interquartile range of NDVI per class, training fields only."""
    cols = [c for c in matrix.columns if c.startswith("NDVI_")]
    dates = [column_date(c) for c in cols]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))
    ax.axvspan(pd.Timestamp("2025-10-01"), pd.Timestamp("2026-05-31"), color="#f4e7c8", alpha=0.6, lw=0,
               label="feature window")
    for cls in CLASSES:
        part = matrix.loc[y.index[y == cls], cols]
        q1, med, q3 = part.quantile(0.25), part.median(), part.quantile(0.75)
        ax.fill_between(dates, q1, q3, color=COLOURS[cls], alpha=0.25, lw=0)
        ax.plot(dates, med, color=COLOURS[cls], lw=1.6, label=f"{cls.replace('_', ' ')} (n={sum(y == cls)})")
    ax.set_ylabel("NDVI")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", ncol=4, frameon=False)
    ax.margins(x=0)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH * 0.85, 2.05))
    for ax, (title, y, pred) in zip(axes, panels):
        ax.spines[:].set_visible(True)
        cm = confusion_matrix(y, pred, labels=CLASSES)
        ax.imshow(cm, cmap="Greens")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        names = [c.replace("_", " ") for c in CLASSES]
        ax.set_xticks(range(3), [c.replace("_", "\n") for c in CLASSES])
        ax.set_yticks(range(3), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("label")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_weights(model, columns, path):
    """Coefficients of the final logistic regression on standardised features, one row per class."""
    clf = model.steps[-1][1]
    order = list(clf.classes_)
    coef = pd.DataFrame(clf.coef_, index=order, columns=columns).loc[CLASSES]
    lim = np.abs(coef.values).max()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.55))
    im = ax.imshow(coef.values, cmap="BrBG", vmin=-lim, vmax=lim, aspect="auto")
    ax.spines[:].set_visible(True)
    ax.set_yticks(range(3), [c.replace("_", " ") for c in CLASSES])
    ax.set_xticks(range(len(columns)), [c.replace("_", " ") for c in columns], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="weight")
    fig.savefig(path)
    plt.close(fig)


def fig_errors(matrix, y_train, test, path):
    """Each misclassified test field's NDVI over the median curves of its labelled and predicted class."""
    cols = [c for c in matrix.columns if c.startswith("NDVI_")]
    dates = [column_date(c) for c in cols]
    wrong = test[test["true"] != test["pred"]]
    fig, axes = plt.subplots(1, len(wrong), figsize=(TEXT_WIDTH, 2.15), sharey=True, squeeze=False)
    for ax, (name, r) in zip(axes[0], wrong.iterrows()):
        for cls, ls in ((r["true"], "-"), (r["pred"], "--")):
            ax.plot(dates, matrix.loc[y_train.index[y_train == cls], cols].median(), color=COLOURS[cls], lw=1.2,
                    ls=ls)
        ax.plot(dates, matrix.loc[name, cols], color="black", lw=1, marker=".", ms=2, label=name)
        ax.set_title(f"{name}: labelled {r['true'].replace('_', ' ')},\npredicted {r['pred'].replace('_', ' ')}")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.set_ylim(0, 0.95)
    axes[0][0].set_ylabel("NDVI")
    classes = sorted(set(wrong["true"]) | set(wrong["pred"]), key=CLASSES.index)
    handles = [plt.Line2D([], [], color=COLOURS[c], lw=1.4, label=f"median {c.replace('_', ' ')}") for c in classes]
    handles += [plt.Line2D([], [], color="grey", lw=1.2, ls="-", label="labelled class"),
                plt.Line2D([], [], color="grey", lw=1.2, ls="--", label="predicted class"),
                plt.Line2D([], [], color="black", lw=1, marker=".", ms=3, label="the field")]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, columnspacing=1.2,
               handlelength=1.8)
    fig.savefig(path)
    plt.close(fig)


# ---------- tables ----------

def tab_data(props, path):
    """Original Rabi 2026 labels, merged classes, training and test counts."""
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Class used & Original label & Train & Test \\", r"\midrule"]
    for cls in CLASSES + ["(left out)"]:
        originals = [k for k, v in MERGE.items() if v == cls] if cls != "(left out)" else ["unknown"]
        for i, orig in enumerate(originals):
            n_train = sum((props["rabi_2026"] == orig) & (props["role"] == "train"))
            n_test = sum((props["rabi_2026"] == orig) & (props["role"] == "test"))
            first = cls.replace("_", r"\_") if i == 0 else ""
            lines.append(f"{first} & {orig.replace('_', chr(92) + '_')} & {n_train} & {n_test} \\\\")
        lines.append(r"\midrule" if cls != "(left out)" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


def tab_models(results, chosen, sizes, path):
    """Balanced accuracy for every model, feature set and CV scheme; the chosen model in bold."""
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Set A (" + str(sizes["A"]) + r")} & \multicolumn{2}{c}{Set B (" + str(sizes["B"]) + r")} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & spatial & random & spatial & random \\", r"\midrule"]
    for name, label in SHORT_NAMES.items():
        cells = []
        for fs in ("A", "B"):
            r = results[(results["features"] == fs) & (results["model"] == name)].iloc[0]
            for col in ("spatial_bal", "random_bal"):
                cell = str(whole(r[col]))
                if (fs, name) == chosen and col == "spatial_bal":
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tab_test(cv, test, path):
    """Per-class recall in spatial CV and on the blind test, with overall scores."""
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Spatial CV} & \multicolumn{2}{c}{Blind test} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Class & $n$ & recall & $n$ & recall \\", r"\midrule"]
    for cls in CLASSES:
        row = [cls.replace("_", r"\_")]
        for d in (cv, test):
            sel = d[d["true"] == cls]
            row += [str(len(sel)), f"{sum(sel['pred'] == cls)}/{len(sel)}"]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    for label, fn in (("Accuracy", acc), ("Balanced acc.", bal)):
        lines.append(f"{label} & & {pct(fn(cv))} & & {pct(fn(test))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def acc(d):
    return float(np.mean(d["true"] == d["pred"]))


def bal(d):
    return float(np.mean([np.mean(d.loc[d["true"] == c, "pred"] == c) for c in CLASSES]))


# ---------- main ----------

def main():
    args = parse_args()
    for sub in ("figures", "generated"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    fig = lambda name: os.path.join(args.out, "figures", name)
    gen = lambda name: os.path.join(args.out, "generated", name)

    cargs = argparse.Namespace(fields=args.fields, matrix=args.matrix, season_start="2025-10-01",
                               season_end="2026-05-31", max_missing=0.5, binary=False)
    matrix, y_all, everyone = load(cargs)
    props = everyone.loc[y_all.index]
    is_train = (props["role"] == "train").values
    y, groups = y_all[is_train], props.loc[is_train, "block"]
    sets = {"A": curve_features(matrix, cargs).loc[y_all.index], "B": hand_features(matrix, cargs).loc[y_all.index]}
    results = pd.read_csv(os.path.join(args.classify, "cv_results.csv"))
    cv = pd.read_csv(os.path.join(args.classify, "cv_predictions.csv"), index_col=0)
    test = pd.read_csv(os.path.join(args.classify, "test_predictions.csv"), index_col=0)
    lc = pd.read_csv(os.path.join(args.classify, "label_confidence.csv"))
    best, top, se, near = choose(results, len(y), {f: X.shape[1] for f, X in sets.items()})
    fs, name = best["features"], best["model"]
    final = fit(models()[name], sets[fs][is_train], y, groups)

    points = gpd.read_file(args.points)
    known = gpd.read_file(args.known)
    merged = everyone["rabi_2026"].map(MERGE).fillna("unknown")
    fig_area(points, known, merged, fig("study_area.pdf"))
    fig_profiles(matrix, y, fig("profiles.pdf"))
    fig_confusion([(f"Spatial CV, training fields (n={len(cv)})", cv["true"], cv["pred"]),
                   (f"Blind test, used once (n={len(test)})", test["true"], test["pred"])], fig("confusion.pdf"))
    if name == "logistic":
        fig_weights(final, list(sets[fs].columns), fig("weights.pdf"))
    fig_errors(matrix, y, test, fig("errors.pdf"))

    tab_data(everyone, gen("tab_data.tex"))
    tab_models(results, (fs, name), {f: X.shape[1] for f, X in sets.items()}, gen("tab_models.tex"))
    tab_test(cv, test, gen("tab_test.tex"))

    obs = pd.read_csv(args.fields_csv)
    centre = known.union_all().centroid
    base_train = float(np.mean(y == y.mode()[0]))
    base_test = float(np.mean(test["true"] == y.mode()[0]))
    t_acc = acc(test)
    conf = test["confidence"]
    high = test[conf == "high"]
    by_source = everyone.loc[test.index]
    jan_apr = [c for c in matrix.columns if c.startswith("NDVI_") and "2026-01-01" <= c[5:] <= "2026-04-30"]
    winter = matrix[jan_apr].mean(axis=1)  # mean NDVI, January to April 2026
    tuned = {k.split("__")[1]: v for k, v in final.get_params().items() if k in models()[name][1]}
    numbers = {
        "NFields": len(everyone), "NTrain": len(y), "NTest": len(test),
        "NBlindLabels": sum(everyone["rabi_2026_source"] == "blind"),
        "NClaudeLabels": sum(everyone["notes"].fillna("").str.startswith("claude (true-colour")),
        "NNotFullyBlind": sum(everyone["notes"].fillna("").str.contains("not fully blind")),
        "NCurveLabels": sum(everyone["rabi_2026_source"] == "curve"),
        "NObs": len(obs), "NFlagged": int((obs["flag"] != "ok").sum()), "NScenes": obs["date"].nunique(),
        "FirstDate": obs["date"].min(), "LastDate": obs["date"].max(),
        "CentreLat": f"{centre.y:.3f}", "CentreLon": f"{centre.x:.3f}",
        "NFeatA": sets["A"].shape[1], "NDatesA": sets["A"].shape[1] // 5, "NFeatB": sets["B"].shape[1],
        "ChosenModel": MODEL_NAMES[name].lower().replace(" (l2)", ""), "ChosenSet": fs,
        "ChosenC": tuned.get("C", "--"), "NNearTop": len(near), "TopBal": pct(top), "SE": f"{100 * se:.1f}\\%",
        "BaselineTrain": pct(base_train), "BaselineTest": pct(base_test),
        "CVAcc": pct(acc(cv)), "CVBal": pct(bal(cv)), "RandomBal": pct(best["random_bal"]),
        "CVCorrect": int((cv["true"] == cv["pred"]).sum()),
        "TestAcc": pct(t_acc), "TestBal": pct(bal(test)), "TestMargin": f"{100 * margin(t_acc, len(test)):.0f}",
        "TestLow": pct(t_acc - margin(t_acc, len(test))),
        "TestCorrect": int((test["true"] == test["pred"]).sum()), "TestBaselineCorrect": int(round(base_test * len(test))),
        "NHigh": len(high), "NHighCorrect": int((high["true"] == high["pred"]).sum()),
        "NTestFromReview": sum(by_source["rabi_2026_source"] == "imagery+curve"),
        "GapAcc": whole(acc(cv)) - whole(t_acc), "GapBal": whole(bal(cv)) - whole(bal(test)),
        "WinterGzero": f"{winter['g0000']:.2f}", "WinterGtwo": f"{winter['g0202']:.2f}",
        "WinterNonCropMax": f"{winter[y.index[y == 'not_cropped']].max():.2f}",
        "NTrainHighMed": int(lc.loc[1, "n_train"]),
        "ConfAllAcc": pct(lc.loc[0, "acc_all"]), "ConfCleanAcc": pct(lc.loc[1, "acc_all"]),
        "ConfAllBal": pct(lc.loc[0, "bal_all"]), "ConfCleanBal": pct(lc.loc[1, "bal_all"]),
    }
    with open(gen("numbers.tex"), "w") as fh:
        fh.write("% generated by s2_paper.py: do not edit by hand\n")
        for k, v in numbers.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Chosen: set {fs} + {name} {tuned}")
    for k, v in numbers.items():
        print(f"  \\{k} = {v}")
    print(f"Saved figures in {args.out}/figures and tables/numbers in {args.out}/generated")


if __name__ == "__main__":
    main()
