"""
Labelling sheets built from true-colour image chips.

--blind (the only mode so far): the test fields, each shown in true colour on 8 dates across
the year, with no NDVI or index anywhere, so their labels are independent of what the model
will learn from. A reference page first shows two fields whose crops are known, to calibrate
the eye. Labels go into labels/blind_test_labels.csv (see LABELS.md for the classes).

    python s2_label_sheets.py --blind

Writes to ./labels/:
    blind_test_00.png         reference fields with their known crops
    blind_test_01.png ...     the unlabelled test fields, 9 per page
    blind_test_labels.csv     the form to fill in (not overwritten if it already exists)
"""
import argparse
import csv
import math
import os
from types import SimpleNamespace

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

import s2_delineate
from s2_fields import read_bounds
from s2_indices import CLEAR_SCL, reflectance

# Months that show each season's story: rice harvest, wheat sowing and growth, wheat ripening and
# harvest, the bare / flooded summer, and the next rice crop.
MONTHS = [(2025, 10), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 6), (2026, 9)]
HALF = 15        # chip half-width in 10 m pixels: 300 m across
PER_PAGE = 9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blind", action="store_true", help="build the blind test sheets")
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--points", default=os.path.join("labels", "sample_points.geojson"))
    p.add_argument("--reference", default="fields.geojson", help="fields with known crops, shown first")
    p.add_argument("--out", default="labels")
    args = p.parse_args()
    if not args.blind:
        p.error("choose a sheet type: --blind")
    return args


def true_colour_stack(points, crs):
    """True colour for each chosen month over the sample area, clouds in grey.

    Uses the same area and monthly scenes as s2_delineate.py, so almost everything is cached.
    """
    area = box(*points.to_crs(crs).total_bounds).buffer(400, join_style="mitre")
    stack_args = SimpleNamespace(start="2025-08-30", end="2026-09-30", flags=os.path.join("output", "fields.csv"),
                                 skip_any_flagged=True)
    _, _, transform, raw = s2_delineate.monthly_stack(area, crs, stack_args)
    by_month = {(it.datetime.year, it.datetime.month): it for it, _, _ in raw}
    frames = []
    for key in MONTHS:
        it = by_month.get(key)
        if it is None:
            continue
        arrays, _ = read_bounds(it, area.bounds, ["B02", "B03", "B04", "SCL"])
        rgb = np.clip(np.dstack([reflectance(arrays[b], it) for b in ("B04", "B03", "B02")]) * 3.5, 0, 1)
        rgb[~np.isin(arrays["SCL"], CLEAR_SCL)] = 0.8
        frames.append((it.datetime.date(), rgb))
    return frames, transform


def chip_row(axes, geom, frames, transform, title):
    """One field across the months: chips centred on the field, its outline in yellow."""
    c = geom.centroid
    col, row = (int(v) for v in ~transform * (c.x, c.y))
    for ax, (day, rgb) in zip(axes, frames):
        r0, c0 = max(row - HALF, 0), max(col - HALF, 0)
        ax.imshow(rgb[r0:row + HALF, c0:col + HALF], interpolation="nearest")
        for poly in getattr(geom, "geoms", [geom]):
            xs, ys = poly.exterior.xy
            cc, rr = ~transform * (np.array(xs), np.array(ys))
            ax.plot(cc - c0 - 0.5, rr - r0 - 0.5, color="yellow", lw=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 2 * HALF - 0.5)
        ax.set_ylim(2 * HALF - 0.5, -0.5)
        ax.set_xlabel(f"{day:%d %b %y}", fontsize=7)
    axes[0].set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)


def page(rows, frames, transform, title, path):
    fig, axes = plt.subplots(len(rows), len(frames), figsize=(1.55 * len(frames) + 1.8, 1.75 * len(rows) + 0.9),
                             squeeze=False)
    for ax_row, (geom, label) in zip(axes, rows):
        chip_row(ax_row, geom, frames, transform, label)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    points = gpd.read_file(args.points)
    crs = points.estimate_utm_crs()
    fields = gpd.read_file(args.fields).to_crs(crs)
    frames, transform = true_colour_stack(points, crs)
    print(f"True colour for {len(frames)} months: " + ", ".join(f"{d:%d %b %y}" for d, _ in frames))

    reference = gpd.read_file(args.reference).to_crs(crs)
    pick = reference[reference["name"].isin(["field1_west", "field1_southeast"])]
    page([(g, f"{n}\nrabi: {r}\nkharif: {k}") for g, n, r, k in
          zip(pick.geometry, pick["name"], pick["rabi_2026"].replace("unknown", "short_winter"), pick["kharif_2026"])],
         frames, transform,
         "Reference fields with known crops. Wheat: green Jan to early Mar, bare by late April (harvested "
         "mid-April). Short winter crop: green around January only, bare by mid-Feb. Rice: bare or flooded "
         "in June, green in September.",
         os.path.join(args.out, "blind_test_00.png"))

    todo = fields[(fields["role"] == "test") & (fields["rabi_2026"].fillna("") == "")].sort_values("name")
    pages = math.ceil(len(todo) / PER_PAGE)
    for k in range(pages):
        chunk = todo.iloc[k * PER_PAGE:(k + 1) * PER_PAGE]
        page([(g, f"{n}\n{a:.1f} ac") for g, n, a in zip(chunk.geometry, chunk["name"], chunk["auto_acres"])],
             frames, transform,
             f"Blind test fields, page {k + 1} of {pages}. Label rabi_2026 and kharif_2026 in "
             "blind_test_labels.csv (classes in LABELS.md); unsure means unknown.",
             os.path.join(args.out, f"blind_test_{k + 1:02d}.png"))

    form = os.path.join(args.out, "blind_test_labels.csv")
    if os.path.exists(form):
        print(f"{form} already exists; not overwritten")
    else:
        with open(form, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "page", "rabi_2026", "kharif_2026", "confidence", "notes"])
            for i, n in enumerate(todo["name"]):
                w.writerow([n, i // PER_PAGE + 1, "", "", "", ""])
    print(f"{len(todo)} test fields on {pages} pages, plus a reference page. Fill in {form}.")


if __name__ == "__main__":
    main()
