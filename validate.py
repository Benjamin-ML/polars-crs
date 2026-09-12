"""Ground-truth validation of polars-crs against pyproj.

Generates REAL coordinates in each system using PROJ (the authoritative
transform library), then checks whether the detector identifies them.

This is the test that matters: the fixtures in tests/ were hand-written and
approximate. These are computed by the same library GIS professionals use.
"""

import numpy as np
import polars as pl
from pyproj import Transformer

import polars_crs as plc

rng = np.random.default_rng(42)

# Real geographic areas for each system, as (lon_min, lon_max, lat_min, lat_max)
AREAS = {
    "EPSG:28992": (3.2, 7.2, 50.75, 53.7),    # Netherlands
    "EPSG:27700": (-7.6, 1.8, 49.9, 60.9),    # Great Britain
    "EPSG:3857": (-180, 180, -85, 85),        # world
    "EPSG:4326": (-180, 180, -90, 90),        # world
}

N = 500
print(f"Validating against pyproj {__import__('pyproj').__version__} "
      f"(PROJ {__import__('pyproj').proj_version_str}), {N} points per system\n")

results = {}
for code, (lon_lo, lon_hi, lat_lo, lat_hi) in AREAS.items():
    lon = rng.uniform(lon_lo, lon_hi, N)
    lat = rng.uniform(lat_lo, lat_hi, N)

    if code == "EPSG:4326":
        x, y = lon, lat
    else:
        tf = Transformer.from_crs("EPSG:4326", code, always_xy=True)
        x, y = tf.transform(lon, lat)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    df = pl.DataFrame({"x": x, "y": y})

    # whole-column verdict
    column_verdict = df.select(plc.detect("x", "y")).item()
    column_candidates = df.select(plc.detect_candidates("x", "y")).item()

    # per-row: how often is the true system among the candidates?
    cands = df.select(plc.candidates("x", "y")).to_series().to_list()
    in_candidates = sum(code in (c or "").split("|") for c in cands) / len(cands)

    guesses = df.select(plc.guess("x", "y")).to_series().to_list()
    exact = sum(g == code for g in guesses) / len(guesses)

    results[code] = (column_verdict, column_candidates, in_candidates, exact)

    ok = "OK  " if column_verdict == code else "MISS"
    print(f"{ok} {code}")
    print(f"       column verdict     {column_verdict}")
    print(f"       column candidates  {column_candidates}")
    print(f"       true CRS in per-row candidates   {in_candidates:6.1%}")
    print(f"       per-row guess exactly right      {exact:6.1%}")
    print()

n_ok = sum(v[0] == k for k, v in results.items())
print(f"Whole-column detection: {n_ok}/{len(results)} correct")
