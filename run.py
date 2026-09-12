"""polars-crs demo: detect the coordinate system of unlabelled x/y columns."""

import polars as pl

import polars_crs as plc

# Four datasets. Nothing tells you what projection any of them are in.
datasets = {
    "mystery_a": pl.DataFrame({"x": [4.9041, -0.1278, -74.0060], "y": [52.3676, 51.5074, 40.7128]}),
    "mystery_b": pl.DataFrame({"x": [121000.0, 92000.0, 233000.0], "y": [487000.0, 437000.0, 582000.0]}),
    "mystery_c": pl.DataFrame({"x": [530000.0, 325000.0, 318000.0], "y": [180000.0, 673000.0, 176000.0]}),
    "mystery_d": pl.DataFrame({"x": [545921.9, -8238310.2], "y": [6866867.1, 4970071.6]}),
}

names = {
    "EPSG:4326": "WGS84 lon/lat",
    "EPSG:3857": "Web Mercator",
    "EPSG:28992": "Dutch RD New",
    "EPSG:27700": "British National Grid",
}

print("Whole-column detection")
print("=" * 62)
for label, df in datasets.items():
    best = df.select(plc.detect("x", "y")).item()
    allc = df.select(plc.detect_candidates("x", "y")).item()
    print(f"  {label}  ->  {best:11}  {names.get(best, ''):22}")
    if allc != best:
        print(f"{'':14}  also consistent with: {allc}")

print()
print("Per-row detail")
print("=" * 62)
detail = datasets["mystery_b"].with_columns(
    guess=pl.col("x").crs.guess("y"),
    candidates=pl.col("x").crs.candidates("y"),
)
print(detail)

print()
print("Honest about ambiguity: RD New and British National Grid overlap.")
print("`guess` takes the narrowest range; `candidates` shows everything.")
