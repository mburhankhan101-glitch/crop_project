"""
Apply the boundary review to the auto-drawn fields: labels/fields_auto.geojson plus
labels/review_decisions.csv -> labels/fields.geojson, the fields to be labelled.

Decisions:
    keep       the outline is one field (or several with the same crop history)
    non_crop   village, road, canal bank or trees; both seasons are labelled non_crop now,
               with source "imagery+curve" because the decision used both
    redraw     the outline was corrected by hand in fields_auto.geojson; kept as edited
    drop       can't tell what is there; removed (avoid it: dropping hard points biases the sample)

Fields not in the decisions file are kept as drawn.

Blind test labels from labels/blind_test_labels.csv (if present) are then merged in, with
source "blind", their confidence per season, and the labeller and reasoning in the notes.

    python s2_review.py
"""
import argparse
import csv
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto", default=os.path.join("labels", "fields_auto.geojson"))
    p.add_argument("--decisions", default=os.path.join("labels", "review_decisions.csv"))
    p.add_argument("--blind", default=os.path.join("labels", "blind_test_labels.csv"))
    p.add_argument("--out", default=os.path.join("labels", "fields.geojson"))
    return p.parse_args()


def main():
    args = parse_args()
    fc = json.load(open(args.auto))
    decisions = {r["name"]: r for r in csv.DictReader(open(args.decisions))}
    unknown = set(decisions) - {f["properties"]["name"] for f in fc["features"]}
    if unknown:
        raise SystemExit(f"{args.decisions} names fields not in {args.auto}: {', '.join(sorted(unknown))}")
    missing = [n for n, r in decisions.items() if r["decision"] not in ("keep", "non_crop", "redraw", "drop")]
    if missing:
        raise SystemExit(f"No valid decision (keep / non_crop / redraw / drop) for: {', '.join(missing)}")

    kept, counts = [], {}
    for f in fc["features"]:
        props = f["properties"]
        r = decisions.get(props["name"])
        decision = r["decision"] if r else "keep"
        counts[decision] = counts.get(decision, 0) + 1
        if decision == "drop":
            continue
        props["review"] = decision if r else ""
        props["review_notes"] = r["notes"] if r else ""
        if decision == "non_crop":
            confidence = "high" if "village" in r["notes"] else "medium"
            for season in ("rabi_2026", "kharif_2026"):
                props[season] = "non_crop"
                props[f"{season}_source"] = "imagery+curve"
                props[f"{season}_confidence"] = confidence
        kept.append(f)

    blind = {r["name"]: r for r in csv.DictReader(open(args.blind))} if os.path.exists(args.blind) else {}
    by_name = {f["properties"]["name"]: f["properties"] for f in kept}
    for name, r in blind.items():
        props = by_name.get(name)
        if props is None:
            raise SystemExit(f"{args.blind} names a field not in {args.out}: {name}")
        for season in ("rabi_2026", "kharif_2026"):
            if r[season]:
                props[season] = r[season]
                props[f"{season}_source"] = "blind"
                props[f"{season}_confidence"] = r[f"{season}_confidence"]
        props["notes"] = f"{r['labeller']}: {r['notes']}"

    with open(args.out, "w") as fh:
        fh.write('{\n  "type": "FeatureCollection",\n  "features": [\n')
        fh.write(",\n".join("    " + json.dumps(f) for f in kept))
        fh.write("\n  ]\n}\n")
    test = [f for f in kept if f["properties"]["role"] == "test"]
    to_label = [f for f in test if not f["properties"]["rabi_2026"]]
    print(f"{len(kept)} fields written to {args.out} (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")")
    print(f"  test fields: {len(test)}, of which {len(to_label)} still need blind labels")
    labelled = [f for f in kept if f["properties"]["rabi_2026"]]
    print(f"  labelled so far: {len(labelled)} fields ({len(blind)} blind, "
          f"{sum(f['properties']['review'] == 'non_crop' for f in kept)} non_crop from the review)")


if __name__ == "__main__":
    main()
