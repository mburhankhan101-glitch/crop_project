# Labelling guide

How the fields in `labels/fields.geojson` are labelled. Every label records where it came from,
so the test set can be kept independent of the data the model learns from.

## What is labelled

- **Unit:** one field polygon (or several neighbouring fields with an identical crop history).
- **Seasons:** `rabi_2026` (winter, Nov 2025 – Apr 2026) and `kharif_2026` (summer, May – Oct 2026).
  Rabi 2026 is the only season the satellite saw from start to finish, so it is the main target.
- **Per label:** the class, a `_source` and a `_confidence` (`high` / `medium` / `low`).

## Classes

| Rabi 2026 | Looks like, in true colour | NDVI profile |
|---|---|---|
| `wheat` | green Jan–Mar, golden in late Mar–Apr, bare after the April harvest | one peak in Feb–Mar, drop in April |
| `short_winter` | green for only part of the winter (e.g. Dec–Jan), bare while neighbouring wheat is still green | a short peak, harvested by mid-Feb or earlier |
| `sugarcane` | green through most of the year, including June when other fields are bare | high most of the year; one long season |
| `orchard` | dark green and textured all year (trees) | steady and fairly high all year |
| `fallow` | brown or beige all winter | stays near bare soil (about 0.1–0.25) |
| `non_crop` | houses, roads, canals, water, tree lines on banks | never above about 0.3, or permanent vegetation that is never harvested |
| `unknown` | you can't tell | |

| Kharif 2026 | Looks like, in true colour | NDVI profile |
|---|---|---|
| `rice` | dark in June–July (flooded, water and mud), bright green Aug–Sep, golden in October | drop to about 0 when flooded (NDWI peak), then a steep rise |
| `other_kharif` | green in summer without the flooded dark phase (maize, fodder, vegetables) | green-up without the flooding signal |
| `sugarcane`, `orchard`, `fallow`, `non_crop`, `unknown` | as above | as above |

## Rules

1. **When unsure, `unknown`.** A missing label costs one field; a wrong one teaches the model a mistake.
2. **Label what is at the point.** A point on a ridge or path belongs to the field it touches;
   a point on a road verge or embankment is `non_crop`.
3. **Deciding from an NDVI profile** (used in the boundary review):
   - never above about 0.3 all year → `non_crop`
   - never below about 0.35, with trees visible → permanent vegetation, `non_crop` or `orchard`
   - a clear crop cycle (below 0.25, then above 0.6) with little spread across the outline → a crop field
   - the point's own profile is flat while the field beside it isn't → the point is on a verge: `non_crop`
4. **One date can lie.** Check the whole season before deciding (a "fallow" field in March had grown a crop in January).

## Sources

| `_source` | Meaning | Used for testing? |
|---|---|---|
| `blind` | labelled from true-colour chips only, before seeing any NDVI for that field | yes |
| `imagery` | high-resolution basemap or Google Earth Pro, for obvious classes | yes |
| `imagery+curve` | decided from both an image and the NDVI profile (the boundary review) | with care |
| `curve` | read from the field's NDVI / index series, or from its k-means cluster | training only |

## The blind test protocol

The 30 test fields (`role: test`, 6 per 2 km strip) are labelled first, from
`labels/blind_test_*.png` only, into `labels/blind_test_labels.csv`. Don't open any NDVI chart or
series output for those fields until their blind labels are saved and committed. Afterwards, the
model's accuracy is reported on these blind labels, separately from the curve-based ones.
