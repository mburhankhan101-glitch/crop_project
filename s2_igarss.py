"""
Step 15: everything the IGARSS 2027 paper needs except the text, in paper_igarss/.

Builds on s2_paper.py (Step 9), at IEEE two-column size:
    figures/        study area, confusion matrices, few-labels curve (3.45 in = one column)
    generated/      numbers.tex (all Step 8 macros), extra.tex (Steps 12-14 and model sizes),
                    ground.tex (field-visit results, or red "pending" marks until the visit is done),
                    tab_eval.tex (CV vs blind vs farmers), tab_models_size.tex (does model size help?)

The text is written by hand in paper_igarss/main.tex. Run again after the field visit
(python s2_ground.py, then python s2_igarss.py) and the ground-truth numbers fill themselves in.

    python s2_igarss.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import s2_paper as sp

OUT = "paper_igarss"
COLUMN = 3.45   # inches: one IEEE column


def whole(x):
    return int(np.floor(100 * x + 0.5 + 1e-9))


def pct(x):
    return f"{whole(x)}\\%"


def params():
    """Weights of each model, counted from the model definitions rather than typed in."""
    from s2_torch import MLP, TempCNN
    from s2_unet import UNet
    count = lambda m: sum(p.numel() for p in m.parameters())
    out = {"logistic": 16 * 3 + 3, "mlp": count(MLP(16)), "tempcnn": count(TempCNN(5, 49)), "unet": count(UNet(16))}
    presto = os.path.join("cache", "external", "presto")
    if os.path.exists(presto):
        sys.path.insert(0, presto)
        from single_file_presto import Presto
        out["presto"] = count(Presto.construct().encoder)
    return out


def fig_few_labels(path):
    curve = pd.read_csv(os.path.join("output", "presto", "few_labels.csv"))
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    style = {"hand-made (16)": ("#c9962b", "hand-made features"), "Presto (128)": ("#356fa8", "Presto embedding"),
             "raw monthly (64)": ("#7a7a7a", "same inputs, no pretraining")}
    for name, (colour, label) in style.items():
        s = curve[curve["features"] == name].groupby("fields_per_fold")["balanced"]
        m, sd = s.mean(), s.std()
        ax.plot(m.index, m.values, "o-", color=colour, ms=3, lw=1.2, label=label)
        ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=0.15, lw=0)
    sizes = sorted(curve["fields_per_fold"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("labelled training fields per fold")
    ax.set_ylabel("balanced accuracy")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(panels, path):
    """Confusion matrices sized for one column: curve labels (CV), blind labels and, after the visit, farmers."""
    from sklearn.metrics import confusion_matrix
    classes = ["wheat", "other_crop", "not_cropped"]
    ticks = ["wheat", "other", "not\ncrop."]
    fig, axes = plt.subplots(1, len(panels), figsize=(COLUMN, 1.55 if len(panels) == 3 else 1.9), squeeze=False)
    for ax, (title, y, pred) in zip(axes[0], panels):
        cm = confusion_matrix(y, pred, labels=classes)
        ax.imshow(cm, cmap="Greens")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), ticks, fontsize=6)
        ax.set_yticks(range(3), ticks if ax is axes[0][0] else [""] * 3, fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.tick_params(length=2)
    axes[0][0].set_ylabel("label", fontsize=7)
    fig.supxlabel("predicted", fontsize=7, y=0.0)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def ground_numbers():
    """Field-visit results from s2_ground.py, or None before the visit."""
    path = os.path.join("output", "ground", "ground_check.csv")
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path, dtype=str).fillna("")
    g = g[g["ground_rabi"] != ""]
    out = {"n": len(g)}
    for role in ("test", "train"):
        sub = g[g["role"] == role]
        for col, key in (("label_rabi", "Labels"), ("step8_model", "Model")):
            d = sub[sub[col] != ""]
            out[f"{role}{key}"] = (int((d["ground_rabi"] == d[col]).sum()), len(d))
    return out


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "generated"), exist_ok=True)
    # Step 9's figures, tables and numbers, drawn at column width.
    sp.TEXT_WIDTH = COLUMN
    argv = sys.argv
    sys.argv = ["s2_paper.py", "--out", OUT]
    sp.main()
    sys.argv = argv
    fig_few_labels(os.path.join(OUT, "figures", "few_labels.pdf"))

    torch_r = pd.read_csv(os.path.join("output", "torch", "cv_results.csv")).set_index("model")
    unet_r = pd.read_csv(os.path.join("output", "unet", "cv_results.csv")).set_index("model")
    presto_r = pd.read_csv(os.path.join("output", "presto", "cv_results.csv")).set_index("features")
    few = pd.read_csv(os.path.join("output", "presto", "few_labels.csv")).groupby(
        ["fields_per_fold", "features"])["balanced"].mean()
    w = params()
    extra = {
        "MlpBal": pct(torch_r.loc["mlp", "field_balanced"]), "TempcnnBal": pct(torch_r.loc["tempcnn", "field_balanced"]),
        "UnetSparseBal": pct(unet_r.loc["sparse", "field_balanced"]),
        "UnetPseudoBal": pct(unet_r.loc["pseudo", "field_balanced"]),
        "PrestoBal": pct(presto_r.loc["Presto (128)", "balanced"]),
        "RawMonthlyBal": pct(presto_r.loc["raw monthly (64)", "balanced"]),
        "FewThreeHand": pct(few[(3, "hand-made (16)")]), "FewThreePresto": pct(few[(3, "Presto (128)")]),
        "FewThreeRaw": pct(few[(3, "raw monthly (64)")]),
        "FewTopHand": pct(few[(few.index.levels[0].max(), "hand-made (16)")]),
        "FewTopPresto": pct(few[(few.index.levels[0].max(), "Presto (128)")]),
        "FewTopRaw": pct(few[(few.index.levels[0].max(), "raw monthly (64)")]),
        "FewTopSize": int(few.index.levels[0].max()),
        **{f"Params{k.capitalize()}": f"{v:,}".replace(",", "{,}") for k, v in w.items()},
    }
    with open(os.path.join(OUT, "generated", "extra.tex"), "w") as fh:
        fh.write("% generated by s2_igarss.py: do not edit by hand\n")
        for k, v in extra.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # Does model size help? One row per model, all on the same spatial CV of the 68 training fields.
    rows = [("Logistic regression, 16 features", w["logistic"], torch_r.loc["logistic", "field_balanced"]),
            ("MLP, 16 features", w["mlp"], torch_r.loc["mlp", "field_balanced"]),
            ("TempCNN, raw 5-day curves", w["tempcnn"], torch_r.loc["tempcnn", "field_balanced"]),
            ("U-Net, sparse labels", w["unet"], unet_r.loc["sparse", "field_balanced"]),
            ("U-Net, + pseudo-labels", w["unet"], unet_r.loc["pseudo", "field_balanced"])]
    if "presto" in w:
        rows.append(("Presto (frozen) + logistic", w["presto"], presto_r.loc["Presto (128)", "balanced"]))
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Model & Weights & Bal.\ acc. \\", r"\midrule"]
    num = lambda n: f"{n:,}".replace(",", "{,}")   # 212{,}547: a comma LaTeX won't space as punctuation
    lines += [f"{name} & {num(n)} & {pct(b)} \\\\" for name, n, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "generated", "tab_models_size.tex"), "w").write("\n".join(lines) + "\n")

    # The three evaluations: curve labels in CV, blind labels, farmers' answers.
    cv = pd.read_csv(os.path.join("output", "classify", "cv_predictions.csv"), index_col=0)
    test = pd.read_csv(os.path.join("output", "classify", "test_predictions.csv"), index_col=0)
    g = ground_numbers()
    panels = [(f"Curve labels, CV (n={len(cv)})", cv["true"], cv["pred"]),
              (f"Blind labels (n={len(test)})", test["true"], test["pred"])]
    gpath = os.path.join("output", "ground", "ground_check.csv")
    if g:
        gc = pd.read_csv(gpath, dtype=str).fillna("")
        gc = gc[(gc["role"] == "test") & (gc["ground_rabi"] != "") & (gc["step8_model"] != "")]
        panels.append((f"Farmers (n={len(gc)})", gc["ground_rabi"], gc["step8_model"]))
    fig_confusion(panels, os.path.join(OUT, "figures", "confusion.pdf"))
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Evaluated against & $n$ & Accuracy \\", r"\midrule",
             f"Curve-derived labels (spatial CV) & {len(cv)} & {pct(sp.acc(cv))} \\\\",
             f"Blind labels from true colour & {len(test)} & {pct(sp.acc(test))} \\\\"]
    if g and g["testModel"][1]:
        k, n = g["testModel"]
        lines.append(f"Farmer-confirmed (test fields) & {n} & {pct(k / n)} \\\\")
    else:
        lines.append(r"Farmer-confirmed (test fields) & \pending{n} & \pending{acc.} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "generated", "tab_eval.tex"), "w").write("\n".join(lines) + "\n")

    with open(os.path.join(OUT, "generated", "ground.tex"), "w") as fh:
        fh.write("% generated by s2_igarss.py from output/ground/ground_check.csv\n")
        if g:
            fh.write(f"\\newcommand{{\\GroundN}}{{{g['n']}}}\n")
            for key, value in g.items():
                if key != "n":
                    fh.write(f"\\newcommand{{\\Ground{key[0].upper() + key[1:]}}}{{{value[0]}/{value[1]}}}\n")
        else:
            for name in ("GroundN", "GroundTestLabels", "GroundTestModel", "GroundTrainLabels", "GroundTrainModel"):
                fh.write(f"\\newcommand{{\\{name}}}{{\\pending{{{name}}}}}\n")
    print(f"Saved figures and generated tables in {OUT}; ground truth: "
          f"{'from ' + str(g['n']) + ' visited fields' if g else 'pending (marked in red)'}")


if __name__ == "__main__":
    main()
