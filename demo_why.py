"""Why this matters: the cost of guessing the coordinate system wrong.

Scenario: a colleague sends you locations.csv. Two numeric columns, no
metadata, no README. You need the distance between the sites.
"""

import math

import polars as pl
from pyproj import Transformer

import polars_crs as plc

# The CSV you were handed. Amsterdam and Rotterdam -- but nothing says so.
df = pl.DataFrame({
    "site": ["site_a", "site_b"],
    "x": [121000.0, 92000.0],
    "y": [487000.0, 437000.0],
})

print("locations.csv")
print(df)
print()

# --- What a junior usually does ---------------------------------------------
# "Two numbers, must be lat/lon." It is the default assumption everywhere.

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

a, b = df.row(0, named=True), df.row(1, named=True)
wrong = haversine(a["y"], a["x"], b["y"], b["x"])

print("ATTEMPT 1 -- assume the columns are lat/lon")
print(f"  haversine says: {wrong:,.1f} km")
print("  No error. No warning. Just a number you can put in a report.")
print()

# --- What polars-crs tells you ----------------------------------------------

verdict = df.select(plc.detect("x", "y")).item()
report = df.select(plc.detect_report("x", "y")).item()

print("ATTEMPT 2 -- ask what the coordinates actually are")
print(f"  plc.detect('x','y')        -> {verdict}")
print(f"  plc.detect_report('x','y') -> {report}")
print(f"  {plc.CRS_RANGES[verdict]['name']}: metres, not degrees.")
print()

# Now the distance can be computed correctly.
tf = Transformer.from_crs(verdict, "EPSG:4326", always_xy=True)
lon1, lat1 = tf.transform(a["x"], a["y"])
lon2, lat2 = tf.transform(b["x"], b["y"])
right = haversine(lat1, lon1, lat2, lon2)

print(f"  reprojected to WGS84: ({lat1:.4f}, {lon1:.4f}) and ({lat2:.4f}, {lon2:.4f})")
print(f"  correct distance: {right:,.1f} km")
print()

print("=" * 66)
print(f"  wrong answer   {wrong:>10,.1f} km")
print(f"  right answer   {right:>10,.1f} km")
print(f"  error          {abs(wrong - right) / right:>10,.0%}")
print("=" * 66)
print()
print("The point is not that the number is wrong. It is that nothing")
print("told you it was wrong. The pipeline ran, the plot rendered, the")
print("dashboard shipped. detect() turns a silent failure into a value")
print("you can assert on.")
