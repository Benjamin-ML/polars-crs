"""Three CSVs, same eight Dutch cities, three different coordinate systems.

None of the files records which system it uses -- exactly like the data you
actually get handed. We detect it, reproject, and compute distances.
"""

import math

import polars as pl
from pyproj import Transformer

import polars_crs as plc

FILES = ["data/cities_a.csv", "data/cities_b.csv", "data/cities_c.csv"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --- step 1: what are these files? ------------------------------------------

print("STEP 1 - three unlabelled files")
print("=" * 72)
detected = {}
for f in FILES:
    df = pl.read_csv(f)
    crs = df.select(plc.detect("x", "y")).item()
    report = df.select(plc.detect_report("x", "y")).item()
    detected[f] = crs
    print(f"  {f}")
    print(f"      first row      {df.row(0)[1]:>16,.2f}  {df.row(0)[2]:>16,.2f}")
    print(f"      detect()       {crs}  ({plc.CRS_RANGES[crs]['name']})")
    print(f"      confidence     {report}")
print()

# --- step 2: distances from Amsterdam, computed from each file --------------

print("STEP 2 - distance from Amsterdam, computed independently from each file")
print("=" * 72)

results = {}
for f, crs in detected.items():
    df = pl.read_csv(f)
    if crs == "EPSG:4326":
        lons, lats = df["x"].to_list(), df["y"].to_list()
    else:
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lons, lats = tf.transform(df["x"].to_list(), df["y"].to_list())
    origin = (lats[0], lons[0])
    results[f] = [haversine_km(*origin, la, lo) for la, lo in zip(lats, lons)]

cities = pl.read_csv(FILES[0])["city"].to_list()
# Spread computed from the FULL-PRECISION values, not the rounded display
# ones -- rounding first would make the agreement look perfect by construction.
spread_m = [
    (max(a, b, c) - min(a, b, c)) * 1000
    for a, b, c in zip(results[FILES[0]], results[FILES[1]], results[FILES[2]])
]

table = pl.DataFrame({
    "city": cities,
    "from_a (4326)": [round(v, 2) for v in results[FILES[0]]],
    "from_b (28992)": [round(v, 2) for v in results[FILES[1]]],
    "from_c (3857)": [round(v, 2) for v in results[FILES[2]]],
    "spread_mm": [round(v * 1000, 3) for v in spread_m],
})
print(table)
print()
print("  Three different source projections, three independent reprojections,")
print(f"  agreement within {max(spread_m) * 1000:.3f} mm over distances up to "
      f"{max(results[FILES[0]]):.0f} km.")
print("  (residual is float rounding in the CSV, not projection error --")
print("   the files store coordinates to 4 decimal places.)")
print()

# --- step 3: what happens without detection ---------------------------------

print("STEP 3 - the same calculation, assuming lat/lon without checking")
print("=" * 72)
for f, crs in detected.items():
    df = pl.read_csv(f)
    xs, ys = df["x"].to_list(), df["y"].to_list()
    try:
        naive = haversine_km(ys[0], xs[0], ys[1], xs[1])
        note = f"{naive:>12,.1f} km"
    except ValueError as e:
        note = f"ERROR: {e}"
    truth = results[f][1]
    flag = "correct" if abs(naive - truth) < 0.1 else f"WRONG - should be {truth:,.1f} km"
    print(f"  {f}  Amsterdam->Rotterdam  {note}   {flag}")

print()
print("  cities_a happens to be right, because it really is lat/lon.")
print("  The other two produce a number with no error and no warning.")
