"""
Spectral signatures: reflectance in all 12 Sentinel-2 L2A bands for a few
hand-picked pixels around Lahore, plotted against wavelength.

    python spectral_signatures.py

Edit TARGETS below to try your own pixels (right-click in Google Maps to copy lat/lon).

Writes to ./output/:
    spectral_signatures.png   one curve per pixel
    spectral_signatures.csv   reflectance per band per pixel
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from s2_indices import reflectance, search

# Sentinel-2 has 13 bands; B10 (cirrus) is only in Level-1C, so Level-2A has these 12.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

SCL_NAMES = {
    0: "no data", 1: "saturated", 2: "dark area", 3: "cloud shadow", 4: "vegetation", 5: "bare soil",
    6: "water", 7: "unclassified", 8: "cloud (medium)", 9: "cloud (high)", 10: "thin cirrus", 11: "snow",
}

# Each pixel was checked against high-resolution imagery and its NDVI / NDWI.
# The river and city use 2 March: on 7 March a small patch of cloud sat right over
# Lahore, even though the scene as a whole was only 4% cloudy.
TARGETS = [
    {"label": "Wheat field", "lat": 31.18985, "lon": 74.29810, "date": "2026-03-07", "color": "#2e7d32"},
    {"label": "Same field after harvest", "lat": 31.18985, "lon": 74.29810, "date": "2026-05-21", "color": "#a1662f"},
    {"label": "Ravi river at Shahdara", "lat": 31.61439, "lon": 74.30390, "date": "2026-03-02", "color": "#1f6fb2"},
    {"label": "Walled City rooftops", "lat": 31.58300, "lon": 74.31300, "date": "2026-03-02", "color": "#555555"},
    {"label": "Cloud over the farmland", "lat": 31.19230, "lon": 74.30190, "date": "2026-01-08", "color": "#9c8fd6"},
]

# Parts of the spectrum, for shading the chart.
REGIONS = [
    (430, 700, "Visible", "#f4efe4"),
    (700, 800, "Red edge", "#f8e1e1"),
    (800, 1000, "Near-infrared", "#efe3f3"),
    (1500, 2300, "Short-wave infrared", "#e3ebf6"),
]


def signature(target):
    """Reflectance of the single pixel under (lat, lon) in every band, on that date."""
    items = search(target["lat"], target["lon"], target["date"], target["date"], max_cloud=101)
    if not items:
        raise SystemExit(f"No Sentinel-2 scene on {target['date']} over {target['label']}")
    item = items[0]

    crs = item.properties.get("proj:code") or f"EPSG:{item.properties['proj:epsg']}"
    xs, ys = warp_transform("EPSG:4326", crs, [target["lon"]], [target["lat"]])
    point = [(xs[0], ys[0])]

    def sample(band):
        # Each band file has its own grid (10, 20 or 60 m), so sample() reads
        # whichever pixel of that grid contains the point.
        with rasterio.open(item.assets[band].href) as src:
            return next(src.sample(point))[0]

    meta = [item.assets[b].extra_fields for b in BANDS]
    return {
        **target,
        "item": item.id,
        # STAC stores centre wavelengths in micrometres.
        "wavelength_nm": np.array([m["eo:bands"][0]["center_wavelength"] * 1000 for m in meta]),
        "gsd_m": [int(m["gsd"]) for m in meta],
        "reflectance": reflectance(np.array([sample(b) for b in BANDS]), item),
        "scl": SCL_NAMES.get(int(sample("SCL")), "unknown"),
    }


def indices(sig):
    r = dict(zip(BANDS, sig["reflectance"]))
    nd = lambda a, b: (r[a] - r[b]) / (r[a] + r[b])
    return {"NDVI": nd("B08", "B04"), "NDWI": nd("B03", "B08"), "NDMI": nd("B8A", "B11")}


def plot(sigs, path):
    fig, ax = plt.subplots(figsize=(13, 7))

    top = max(s["reflectance"].max() for s in sigs) * 1.12
    for lo, hi, name, color in REGIONS:
        ax.axvspan(lo, hi, color=color, lw=0, zorder=0)
        ax.text((lo + hi) / 2, top * 0.97, name, ha="center", va="top", fontsize=9, color="#666")
    ax.text(1250, top * 0.8, "No bands here:\nwater vapour in the\natmosphere absorbs\nthese wavelengths",
            ha="center", va="center", fontsize=8, color="#999", style="italic")

    for s in sigs:
        wl, refl = s["wavelength_nm"], s["reflectance"]
        ax.plot(wl, refl, "-", color=s["color"], lw=2, zorder=2,
                label=f"{s['label']}  ·  {s['date']}  ·  SCL: {s['scl']}")
        for x, y, gsd in zip(wl, refl, s["gsd_m"]):
            # Filled = 10 m, smaller filled = 20 m, hollow = 60 m atmospheric band.
            if gsd == 60:
                ax.scatter(x, y, s=40, facecolor="white", edgecolor=s["color"], lw=1.5, zorder=3)
            else:
                ax.scatter(x, y, s=55 if gsd == 10 else 30, color=s["color"], zorder=3)

    wl = sigs[0]["wavelength_nm"]
    band_axis = ax.secondary_xaxis("top")
    band_axis.set_xticks(wl, labels=BANDS, rotation=90, fontsize=8)
    band_axis.tick_params(length=3)

    ax.set_xlim(400, 2300)
    ax.set_ylim(0, top)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Surface reflectance (fraction of sunlight reflected)")
    ax.set_title("Spectral signatures around Lahore, Sentinel-2 L2A\n"
                 "large dot = 10 m band · small dot = 20 m · hollow = 60 m atmospheric band", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("output", exist_ok=True)
    print(f"Sampling {len(BANDS)} bands at {len(TARGETS)} pixels ...")
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as pool:
        sigs = list(pool.map(signature, TARGETS))

    print(f"\n{'pixel':26} {'date':11} {'SCL class':15} {'NDVI':>6} {'NDWI':>6} {'NDMI':>6}")
    for s in sigs:
        i = indices(s)
        print(f"{s['label']:26} {s['date']:11} {s['scl']:15} {i['NDVI']:+6.2f} {i['NDWI']:+6.2f} {i['NDMI']:+6.2f}")

    csv_path = os.path.join("output", "spectral_signatures.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "wavelength_nm", "pixel_m"] + [f"{s['label']} ({s['date']})" for s in sigs])
        for k, band in enumerate(BANDS):
            w.writerow([band, round(sigs[0]["wavelength_nm"][k]), sigs[0]["gsd_m"][k]]
                       + [f"{s['reflectance'][k]:.4f}" for s in sigs])

    png_path = os.path.join("output", "spectral_signatures.png")
    plot(sigs, png_path)
    print(f"\nSaved {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
