"""Plugin vs the pure-Python equivalent."""
import time
import numpy as np
import polars as pl
import polars_crs as plc

N = 2_000_000
rng = np.random.default_rng(0)
df = pl.DataFrame({
    "x": rng.uniform(0, 300_000, N),
    "y": rng.uniform(289_000, 629_000, N),
})

# Read the ranges from the package so this reference cannot drift out of sync
# with the Rust definitions -- it did once, and the mismatch showed up as
# "results differ!" rather than as a wrong benchmark.
RANGES = [
    (code, r["x"][0], r["x"][1], r["y"][0], r["y"][1])
    for code, r in plc.CRS_RANGES.items()
]

def py_guess(row):
    x, y = row["x"], row["y"]
    for code, xn, xx, yn, yx in RANGES:
        if xn <= x <= xx and yn <= y <= yx:
            return code
    return "unknown"

t = time.perf_counter()
a = df.with_columns(g=plc.guess("x", "y"))
plugin = time.perf_counter() - t

t = time.perf_counter()
b = df.with_columns(
    g=pl.struct("x", "y").map_elements(py_guess, return_dtype=pl.String)
)
python = time.perf_counter() - t

assert a["g"].to_list() == b["g"].to_list(), "results differ!"
print(f"  rows           {N:,}")
print(f"  plugin (Rust)  {plugin:7.3f} s")
print(f"  map_elements   {python:7.3f} s")
print(f"  speedup        {python/plugin:7.1f}x")
