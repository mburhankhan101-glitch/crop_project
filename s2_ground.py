"""
Score everything against field-visit ground truth (labels/ground_truth.csv, filled from field_kit).

Ground truth is independent of all labels and models, so this is the most trustworthy check in the
project. For every visited field with a farmer-confirmed answer it compares:

    Rabi 2026 (farmer's answer about last winter) with
        the label used so far (blind/AI, curve or review), the Step 8 model (CV or the blind test),
        the Step 10 crop map and the Step 13 U-Net map (majority of the pixels in the outline)
    Kharif 2026 (what is growing now, October) with the Kharif 2026 label

    python s2_ground.py                     # all answers
    python s2_ground.py --only-sure         # only answers the farmer was sure of

Outputs in output/ground/: ground_check.csv (one row per field) and printed agreement tables.
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from s2_classify import MERGE

# Farmer's words -> the three Rabi classes the models predict.
RABI_GROUND = {"wheat": "wheat", "mustard": "other_crop", "potato": "other_crop", "berseem": "other_crop",
               "fodder": "other_crop", "vegetables": "other_crop", "sugarcane": "other_crop", "orchard": "other_crop",
               "gram": "other_crop", "barley": "other_crop", "fallow": "not_cropped", "non_crop": "not_cropped"}
NON_CROP_USE = {"village", "road", "canal", "trees", "water", "brick_kiln"}
KHARIF_GROUND = {"rice": "rice", "rice_harvested": "rice", "sugarcane": "sugarcane", "cotton": "other_kharif",
                 "maize": "other_kharif", "fodder": "other_kharif", "vegetables": "other_kharif", "bare": "fallow"}
MAP_CODES = {"crop_map.tif": {1: "wheat", 2: "other_crop", 3: "not_cropped", 4: "uncertain"},
             "crop_map_unet.tif": {1: "not_cropped", 2: "other_crop", 3: "wheat", 4: "uncertain"}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ground", default=os.path.join("labels", "ground_truth.csv"))
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--only-sure", action="store_true")
    p.add_argument("--out", default=os.path.join("output", "ground"))
    return p.parse_args()


def clean(v):
    return str(v).strip().lower().replace(" ", "_") if pd.notna(v) else ""


def ground_rabi(r):
    use, last = clean(r["land_use"]), clean(r["crop_last_winter"])
    if use in NON_CROP_USE:
        return "not_cropped"           # a village or road was not cropped last winter either
    return RABI_GROUND.get(last, "")   # "" = unknown (dont_know, blank or an unlisted word)


def map_majority(path, codes, geom_by_field):
    with rasterio.open(path) as src:
        m = src.read(1)
        out = {}
        for name, geom in geom_by_field.items():
            inside = ~geometry_mask([mapping(geom.buffer(-5))], out_shape=m.shape, transform=src.transform)
            vals = m[inside]
            vals = vals[vals > 0]
            out[name] = codes.get(int(np.bincount(vals).argmax()), "") if vals.size else ""
    return out


def agreement(df, truth, cols):
    rows = []
    for c in cols:
        d = df[(df[truth] != "") & (df[c] != "")]
        if len(d):
            rows.append({"compared": c, "fields": len(d), "agree": int((d[truth] == d[c]).sum()),
                         "share": round(float(np.mean(d[truth] == d[c])), 3)})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not os.path.exists(args.ground):
        raise SystemExit(f"{args.ground} not found: fill field_kit/ground_truth_form.csv during the visit and save it there")
    os.makedirs(args.out, exist_ok=True)
    g = pd.read_csv(args.ground, dtype=str)
    g = g[g["visited"].map(clean) == "yes"]
    if args.only_sure:
        g = g[g["farmer_sure"].map(clean) == "sure"]
    fields = gpd.read_file(args.fields)
    props = fields.set_index("name")
    df = pd.DataFrame({"field": g["field"].str.strip()})
    df["role"] = df["field"].map(props["role"])
    df["ground_rabi"] = [ground_rabi(r) for _, r in g.iterrows()]
    df["ground_kharif"] = [("non_crop" if clean(r["land_use"]) in NON_CROP_USE else KHARIF_GROUND.get(clean(r["crop_now"]), ""))
                           for _, r in g.iterrows()]
    df["label_rabi"] = df["field"].map(props["rabi_2026"]).map(lambda v: MERGE.get(v, ""))
    df["label_source"] = df["field"].map(props["rabi_2026_source"])
    df["label_kharif"] = df["field"].map(props["kharif_2026"]).fillna("")

    # Step 8 predictions: cross-validation for training fields, the one blind test for test fields.
    step8 = pd.concat([pd.read_csv(os.path.join("output", "classify", f), index_col=0)["pred"]
                       for f in ("cv_predictions.csv", "test_predictions.csv")])
    df["step8_model"] = df["field"].map(step8).fillna("")
    crs_fields = None
    for name, col in (("crop_map.tif", "step10_map"), ("crop_map_unet.tif", "step13_unet")):
        folder = "cube" if name == "crop_map.tif" else "unet"
        path = os.path.join("output", folder, name)
        if os.path.exists(path):
            with rasterio.open(path) as src:
                crs_fields = fields.to_crs(src.crs).set_index("name").geometry
            maj = map_majority(path, MAP_CODES[name], {f: crs_fields[f] for f in df["field"] if f in crs_fields.index})
            df[col] = df["field"].map(maj).fillna("")
    df.to_csv(os.path.join(args.out, "ground_check.csv"), index=False)

    print(f"{len(df)} visited fields{' (farmer sure only)' if args.only_sure else ''}; "
          f"Rabi ground truth for {int((df['ground_rabi'] != '').sum())}, Kharif for {int((df['ground_kharif'] != '').sum())}")
    cols = [c for c in ("label_rabi", "step8_model", "step10_map", "step13_unet") if c in df]
    for role in ("test", "train"):
        sub = df[df["role"] == role]
        if len(sub):
            print(f"\nRabi 2026, {role} fields: agreement with the farmers")
            print(agreement(sub, "ground_rabi", cols).to_string(index=False))
    print("\nWhich labels held up, by how they were made:")
    by_src = df[(df["ground_rabi"] != "") & (df["label_rabi"] != "")].groupby("label_source").apply(
        lambda d: pd.Series({"fields": len(d), "agree": int((d["ground_rabi"] == d["label_rabi"]).sum())}),
        include_groups=False)
    print(by_src.to_string())
    k = df[(df["ground_kharif"] != "") & (df["label_kharif"] != "") & (df["label_kharif"] != "unknown")]
    if len(k):
        print(f"\nKharif 2026: labels agree with what is growing now on {int((k['ground_kharif'] == k['label_kharif']).sum())} "
              f"of {len(k)} fields")
    wrong = df[(df["ground_rabi"] != "") & (df["label_rabi"] != "") & (df["ground_rabi"] != df["label_rabi"])]
    if len(wrong):
        print("\nRabi labels the farmers contradict:")
        print(wrong[["field", "role", "label_source", "label_rabi", "ground_rabi"]].to_string(index=False))
    print(f"\nSaved ground_check.csv in {args.out}")


if __name__ == "__main__":
    main()
