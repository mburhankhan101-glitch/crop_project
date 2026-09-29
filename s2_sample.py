"""
Grid sample of points to label, centred on the fields in fields.geojson.

One random point in every 1 km cell of a 10 x 10 km square, so the sample is spread evenly
and nobody chooses which fields get in (that would bias it towards big, clear, easy fields).
Points landing on roads, villages or canals are kept: the map needs those classes too.

Each point also gets:
    block   the 2 km north-south strip it falls in, used to hold out whole strips in
            spatial cross-validation, so test fields never have near neighbours in training
    role    "test" for 6 random points per strip (30 in all), to be labelled blind from
            true-colour images only; "train" for the rest

    python s2_sample.py
    python s2_sample.py --size 10000 --cell 1000 --test-per-block 6 --seed 2026

Writes to ./labels/:
    sample_points.geojson   the points, with name, block, role and sample="grid"
    sample_points.kml       the same points for Google Earth Pro (dated imagery) or a phone
"""
import argparse
import json
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default="fields.geojson", help="the sample square is centred on these fields")
    p.add_argument("--size", type=int, default=10_000, help="side of the square in metres (default 10000)")
    p.add_argument("--cell", type=int, default=1_000, help="one point per cell of this size (default 1000 m)")
    p.add_argument("--block-cells", type=int, default=2, help="strip width in cells for spatial CV (default 2)")
    p.add_argument("--test-per-block", type=int, default=6, help="blind test points per strip (default 6)")
    p.add_argument("--seed", type=int, default=2026, help="random seed, so the sample can be reproduced")
    p.add_argument("--out", default="labels", help="output folder")
    return p.parse_args()


def write_kml(points, path):
    """Minimal KML: one placemark per point, test points in a different colour."""
    styles = {"test": "ff0000ff", "train": "ff00ff00"}  # KML colours are aabbggrr: red, green
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>crop_project sample points</name>']
    for role, colour in styles.items():
        lines.append(f'<Style id="{role}"><IconStyle><color>{colour}</color><scale>0.8</scale></IconStyle></Style>')
    for _, p in points.iterrows():
        lines.append(f'<Placemark><name>{p["name"]}</name><styleUrl>#{p["role"]}</styleUrl>'
                     f'<description>{p["role"]}, {p["block"]}</description>'
                     f'<Point><coordinates>{p.geometry.x:.6f},{p.geometry.y:.6f},0</coordinates></Point></Placemark>')
    lines.append("</Document></kml>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    fields = gpd.read_file(args.fields)
    crs = fields.estimate_utm_crs()
    centre = fields.to_crs(crs).union_all().centroid
    x0, y1 = centre.x - args.size / 2, centre.y + args.size / 2
    n = args.size // args.cell

    rows = []
    for r in range(n):          # rows from north to south
        for c in range(n):      # columns from west to east
            x = x0 + (c + rng.random()) * args.cell
            y = y1 - (r + rng.random()) * args.cell
            rows.append({"name": f"g{r:02d}{c:02d}", "block": f"strip_{c // args.block_cells + 1}",
                         "sample": "grid", "geometry": Point(x, y)})
    points = gpd.GeoDataFrame(rows, crs=crs)

    points["role"] = "train"
    for _, members in points.groupby("block"):
        test = rng.choice(members.index, size=min(args.test_per_block, len(members)), replace=False)
        points.loc[test, "role"] = "test"

    points = points.to_crs("EPSG:4326")
    os.makedirs(args.out, exist_ok=True)
    features = [{"type": "Feature",
                 "properties": {k: p[k] for k in ("name", "block", "role", "sample")},
                 "geometry": {"type": "Point", "coordinates": [round(p.geometry.x, 7), round(p.geometry.y, 7)]}}
                for _, p in points.iterrows()]
    with open(os.path.join(args.out, "sample_points.geojson"), "w") as f:
        f.write('{\n  "type": "FeatureCollection",\n  "features": [\n')
        f.write(",\n".join("    " + json.dumps(ft) for ft in features))
        f.write("\n  ]\n}\n")
    write_kml(points, os.path.join(args.out, "sample_points.kml"))

    lon0, lat0, lon1, lat1 = points.total_bounds
    print(f"{len(points)} points, one per {args.cell} m cell, in {args.size / 1000:g} x {args.size / 1000:g} km "
          f"around {centre.x:.0f}, {centre.y:.0f} ({crs.to_string()})")
    print(f"  spans {lat0:.4f}-{lat1:.4f} N, {lon0:.4f}-{lon1:.4f} E")
    print("  per strip: " + ", ".join(f"{b} {len(g)} ({(g.role == 'test').sum()} test)"
                                      for b, g in points.groupby("block")))
    print(f"Saved sample_points.geojson and sample_points.kml in ./{args.out}/")


if __name__ == "__main__":
    main()
