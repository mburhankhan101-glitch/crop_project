"""
Field-visit kit for on-ground validation of the labelled fields near Raiwind.

Priority A (visit all): the 30 blind test fields, plus g0608, the training field every model gets wrong.
Priority B (if time allows): the training fields whose Rabi 2026 label has low confidence.

Each list is put in a short driving order (nearest neighbour, then 2-opt on straight-line distance).
The sheets you carry show NO labels and NO model predictions, so what you record stays independent.

    python s2_fieldkit.py

Outputs in field_kit/:
    visit_points.kml        open in Google Earth (or import into Google My Maps): points + outlines
    visit_points.csv        order, field, coordinates, a Google Maps link per point
    ground_truth_form.csv   one row per point, to fill in (Excel or Google Sheets) during the visit
    field_sheet.html        printable sheet: route maps, the interview questions, a table to write in
    route_A.png, route_B.png  the routes on a March 2026 true-colour image
After the visit, save the filled form as labels/ground_truth.csv and run s2_ground.py.
"""
import argparse
import html
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import s2_cube

FORM_COLUMNS = ["order", "field", "visited", "date_time", "your_lat", "your_lon", "land_use", "crop_now",
                "crop_last_winter", "farmer_answered", "farmer_sure", "outline_matches", "photos", "notes"]
ALLOWED = {
    "visited": "yes / no (couldn't reach, no permission)",
    "land_use": "crop / orchard / fallow / village / road / canal / trees / water / brick_kiln / other",
    "crop_now": "rice / rice_harvested / sugarcane / cotton / maize / fodder / vegetables / bare / other (write it)",
    "crop_last_winter": "wheat / mustard / potato / berseem / fodder / vegetables / sugarcane / orchard / fallow / non_crop / dont_know",
    "farmer_answered": "yes / no (no one there)",
    "farmer_sure": "sure / unsure",
    "outline_matches": "yes / partly (field is bigger or split) / no",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", default=os.path.join("labels", "fields.geojson"))
    p.add_argument("--points", default=os.path.join("labels", "sample_points.geojson"))
    p.add_argument("--extra", default="g0608", help="comma-separated training fields to add to priority A")
    p.add_argument("--rgb-date", default="2026-03-07")
    p.add_argument("--out", default="field_kit")
    return p.parse_args()


def tour(xy):
    """Visiting order: nearest neighbour from the westernmost point, then 2-opt until no swap shortens it."""
    n = len(xy)
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    order, left = [int(np.argmin(xy[:, 0]))], set(range(n))
    left.discard(order[0])
    while left:
        nxt = min(left, key=lambda j: d[order[-1], j])
        order.append(nxt)
        left.discard(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, e = order[j], order[j + 1] if j + 1 < n else None
                before = d[a, b] + (d[c, e] if e is not None else 0)
                after = d[a, c] + (d[b, e] if e is not None else 0)
                if after < before - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    length = sum(d[order[k], order[k + 1]] for k in range(n - 1))
    return order, length


def maps_link(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def write_kml(groups, fields_ll, path):
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           "<name>Raiwind field visit</name>",
           '<Style id="A"><IconStyle><color>ff0055ff</color><scale>1.1</scale></IconStyle>'
           '<LineStyle><color>ff0055ff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>',
           '<Style id="B"><IconStyle><color>ffffaa00</color></IconStyle>'
           '<LineStyle><color>ffffaa00</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>']
    for key, df in groups.items():
        out.append(f"<Folder><name>Priority {key}</name>")
        for _, r in df.iterrows():
            ring = fields_ll.loc[r["field"]].exterior.coords
            coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in ring)
            out.append(f"<Placemark><name>{key}{r['order']:02d} {r['field']}</name><styleUrl>#{key}</styleUrl>"
                       f"<description>{html.escape(r['maps_link'])}</description>"
                       f"<Point><coordinates>{r['lon']:.6f},{r['lat']:.6f},0</coordinates></Point></Placemark>")
            out.append(f"<Placemark><name>{r['field']} outline</name><styleUrl>#{key}</styleUrl>"
                       f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates>"
                       f"</LinearRing></outerBoundaryIs></Polygon></Placemark>")
        out.append("</Folder>")
    out.append("</Document></kml>")
    open(path, "w", encoding="utf-8").write("\n".join(out))


def true_colour(args):
    """The square in true colour on one clear date, from the cached Step 10 blocks (None if not cached)."""
    cargs = s2_cube.parse_args([])
    crs, bounds = s2_cube.square(cargs)
    items = s2_cube.search_geometry(s2_cube.lonlat_polygon(crs, bounds), cargs.start, cargs.end, cargs.max_cloud)
    day = (date.fromisoformat(args.rgb_date) - date(2025, 1, 1)).days
    n = cargs.size // 10
    rgb = np.zeros((n, n, 3), dtype="float32")
    halo = cargs.cloud_buffer
    for r in range(0, n, cargs.block):
        for c in range(0, n, cargs.block):
            rows, cols = slice(r, min(r + cargs.block, n)), slice(c, min(c + cargs.block, n))
            if not os.path.exists(s2_cube.block_path(rows, cols, cargs)):
                return None, crs, bounds
            raw, days = s2_cube.load_block(items, crs, bounds, rows, cols, halo, cargs)
            if day not in days:
                return None, crs, bounds
            t = int(np.flatnonzero(days == day)[0])
            top, left = rows.start - max(rows.start - halo, 0), cols.start - max(cols.start - halo, 0)
            h, w = rows.stop - rows.start, cols.stop - cols.start
            for i, b in enumerate(("B04", "B03", "B02")):
                rgb[rows, cols, i] = raw[b][t, top:top + h, left:left + w]
    return np.clip((rgb - 1000) / 10000 / 0.2, 0, 1), crs, bounds


def plot_route(df, rgb, crs, bounds, title, path):
    x0, y0, x1, y1 = bounds
    pts = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(crs)
    xs, ys = (pts.x - x0) / 1000, (pts.y - y0) / 1000
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    if rgb is not None:
        ax.imshow(rgb, extent=(0, (x1 - x0) / 1000, 0, (y1 - y0) / 1000))
    ax.plot(xs, ys, "-", color="white", lw=1.2, alpha=0.9)
    for (_, r), x, y in zip(df.iterrows(), xs, ys):
        ax.plot(x, y, "o", color="#ff5500", ms=7, mec="black", mew=0.6)
        ax.annotate(f"{r['order']}", (x, y), xytext=(4, 3), textcoords="offset points", color="white",
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, pil_kwargs={"optimize": True})
    plt.close(fig)


def write_sheet(groups, lengths, args, path):
    q = [("What is growing in this field now?", "Is khait mein abhi kya laga hua hai?"),
         ("What did you grow here last winter (sowed around Nov 2025, harvested around Apr 2026)?",
          "Pichli sardiyon mein (Nov 2025 se April 2026 tak) is khait mein kya laga tha?"),
         ("Are you sure? Is this your field, or do you know who farms it?",
          "Kya aap ko yaqeen hai? Kya ye aap ka khait hai?"),
         ("Did that crop cover the whole field, or only part of it?", "Kya poore khait mein yahi fasal thi ya sirf kuch hisse mein?")]
    rows_html = []
    for key, df in groups.items():
        rows_html.append(f'<h2>Priority {key}: {len(df)} points, about {lengths[key]:.0f} km in a straight line between them</h2>')
        rows_html.append('<table><tr><th>#</th><th>Field</th><th>Coordinates</th><th>Visited</th><th>Land use / crop now</th>'
                         '<th>Last winter (farmer)</th><th>Sure?</th><th>Outline OK?</th><th>Photos</th><th>Notes</th></tr>')
        for _, r in df.iterrows():
            rows_html.append(f"<tr><td>{key}{r['order']}</td><td>{r['field']}</td>"
                             f"<td><a href=\"{html.escape(r['maps_link'])}\">{r['lat']:.5f}, {r['lon']:.5f}</a></td>"
                             + "<td></td>" * 7 + "</tr>")
        rows_html.append("</table>")
        rows_html.append(f'<img src="route_{key}.png" alt="Route {key}">')
    allowed = "".join(f"<li><b>{k}</b>: {html.escape(v)}</li>" for k, v in ALLOWED.items())
    questions = "".join(f"<li>{html.escape(e)}<br><i>{html.escape(u)}</i></li>" for e, u in q)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Field visit sheet</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:13px/1.45 system-ui,sans-serif;margin:18px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:8px 0 14px}}td,th{{border:1px solid #888;padding:5px 4px;vertical-align:top}}
td{{height:30px}}th{{background:#eee;font-size:12px}}img{{max-width:100%;margin-bottom:18px}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:22px}}@media print{{h2{{page-break-before:always}}a{{color:#111}}}}</style></head><body>
<h1>Field visit: ground truth near Raiwind</h1>
<p>Visit every Priority A point; Priority B only if time allows. At each point: stand at the coordinates (tap the
link to open Google Maps), check you are on the outlined field, take 4 photos (N, E, S, W), then ask the farmer.
<b>This sheet deliberately shows no labels or model predictions</b>, so that what you record stays independent.</p>
<h2 style="page-break-before:avoid">Questions for the farmer</h2><ol>{questions}</ol>
<h2 style="page-break-before:avoid">Words to use in the form</h2><ul>{allowed}</ul>
<h2 style="page-break-before:avoid">Safety and courtesy</h2><ul><li>Go in daylight, ideally with someone local.</li>
<li>Ask before walking into a field; explain you are a student doing research on crops from satellites.</li>
<li>Don't enter if refused, and don't trample crops: observe from the edge.</li>
<li>Write "dont_know" rather than guessing: an honest blank is better than a wrong label.</li></ul>
{''.join(rows_html)}
<p>After the visit: copy the answers into ground_truth_form.csv, save it as labels/ground_truth.csv, and run
<code>python s2_ground.py</code>.</p></body></html>"""
    open(path, "w", encoding="utf-8").write(page)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    fields = gpd.read_file(args.fields)
    points = gpd.read_file(args.points).set_index("name")
    props = fields.set_index("name")
    extra = [f for f in args.extra.split(",") if f]
    a = sorted(set(props.index[props["role"] == "test"]) | set(extra))
    b = sorted(props.index[(props["role"] == "train") & (props["rabi_2026_confidence"] == "low")
                           & ~props.index.isin(a)])

    crs = fields.estimate_utm_crs()
    utm = points.to_crs(crs)
    groups, lengths = {}, {}
    for key, names in (("A", a), ("B", b)):
        xy = np.c_[utm.loc[names].geometry.x, utm.loc[names].geometry.y]
        order, length = tour(xy)
        names = [names[i] for i in order]
        lat, lon = points.loc[names].geometry.y.to_numpy(), points.loc[names].geometry.x.to_numpy()
        groups[key] = pd.DataFrame({"order": range(1, len(names) + 1), "field": names, "lat": lat.round(6),
                                    "lon": lon.round(6), "maps_link": [maps_link(y, x) for y, x in zip(lat, lon)],
                                    "acres": props.loc[names, "auto_acres"].to_numpy()})
        lengths[key] = length / 1000
    visit = pd.concat([g.assign(priority=k) for k, g in groups.items()])
    visit[["priority", "order", "field", "lat", "lon", "acres", "maps_link"]].to_csv(
        os.path.join(args.out, "visit_points.csv"), index=False)
    form = visit[["priority", "order", "field"]].copy()
    form["order"] = form["priority"] + form["order"].astype(str)
    form = form.drop(columns="priority")
    for c in FORM_COLUMNS[2:]:
        form[c] = ""
    form.to_csv(os.path.join(args.out, "ground_truth_form.csv"), index=False)
    write_kml(groups, fields.set_index("name").geometry, os.path.join(args.out, "visit_points.kml"))

    rgb, sq_crs, bounds = true_colour(args)
    for key, df in groups.items():
        plot_route(df, rgb, sq_crs, bounds, f"Priority {key}: {len(df)} points in visiting order",
                   os.path.join(args.out, f"route_{key}.png"))
    write_sheet(groups, lengths, args, os.path.join(args.out, "field_sheet.html"))
    print(f"Priority A: {len(a)} points ({lengths['A']:.1f} km straight-line route); "
          f"Priority B: {len(b)} points ({lengths['B']:.1f} km)")
    print(f"Saved visit_points.kml, visit_points.csv, ground_truth_form.csv, field_sheet.html, route_A.png, "
          f"route_B.png in {args.out}")


if __name__ == "__main__":
    main()
