"""Five implementations of the same function, compared.

The comparison that matters is against native Polars expressions, which can
express a range check without any Rust. map_elements and a pure Python loop are
included for scale.
"""

import time

import numpy as np
import polars as pl

import polars_crs as plc
from gendata import make

RANGES = [(c, r["x"][0], r["x"][1], r["y"][0], r["y"][1])
          for c, r in plc.CRS_RANGES.items()]


# Each implementation returns a Series or numpy array rather than a Python
# list. Building a 10M-element list costs the same everywhere and would swamp
# the compute being measured.


# --- 1. the plugin -----------------------------------------------------------
def impl_plugin(df):
    return df.select(plc.guess("x", "y")).to_series()


# --- 2. native Polars expressions (no plugin, no Rust of your own) -----------
def impl_polars_native(df):
    expr = pl.lit("unknown")
    for code, xn, xx, yn, yx in reversed(RANGES):   # reversed: first wins
        expr = (
            pl.when(pl.col("x").is_between(xn, xx) & pl.col("y").is_between(yn, yx))
            .then(pl.lit(code))
            .otherwise(expr)
        )
    return df.select(expr).to_series()


# --- 3. numpy ----------------------------------------------------------------
def impl_numpy(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    out = np.full(len(x), "unknown", dtype=object)
    assigned = np.zeros(len(x), dtype=bool)
    for code, xn, xx, yn, yx in RANGES:
        m = (x >= xn) & (x <= xx) & (y >= yn) & (y <= yx) & ~assigned
        out[m] = code
        assigned |= m
    return out


# --- 4. map_elements (row-wise Python inside Polars) -------------------------
def _py_guess(row):
    x, y = row["x"], row["y"]
    for code, xn, xx, yn, yx in RANGES:
        if xn <= x <= xx and yn <= y <= yx:
            return code
    return "unknown"


def impl_map_elements(df):
    return df.select(
        pl.struct("x", "y").map_elements(_py_guess, return_dtype=pl.String)
    ).to_series()


# --- 5. a plain Python loop, no Polars at all --------------------------------
def impl_pure_python(df):
    xs = df["x"].to_list()
    ys = df["y"].to_list()
    out = []
    for x, y in zip(xs, ys):
        label = "unknown"
        for code, xn, xx, yn, yx in RANGES:
            if xn <= x <= xx and yn <= y <= yx:
                label = code
                break
        out.append(label)
    return out


IMPLS = [
    ("plugin (Rust)", impl_plugin),
    ("polars native expr", impl_polars_native),
    ("numpy", impl_numpy),
    ("map_elements", impl_map_elements),
    ("pure python loop", impl_pure_python),
]

SIZES = [100_000, 1_000_000, 10_000_000]
SLOW = {"map_elements", "pure python loop"}
SLOW_LIMIT = 1_000_000   # don't run the slow ones on 10M, it takes minutes


def timed(fn, df, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(df)
        best = min(best, time.perf_counter() - t)
    return best, out


print(f"polars {pl.__version__} | release build | best of 3\n")

for n in SIZES:
    df = make(n)
    print(f"--- {n:,} rows " + "-" * 46)
    baseline = None
    reference = None
    for name, fn in IMPLS:
        if name in SLOW and n > SLOW_LIMIT:
            print(f"  {name:22} {'skipped (too slow)':>14}")
            continue
        secs, out = timed(fn, df)
        as_list = out.to_list() if hasattr(out, "to_list") else list(out)
        if reference is None:
            reference, baseline = as_list, secs
        else:
            assert as_list == reference, f"{name} disagrees with the plugin!"
        rel = baseline / secs
        bar = "x" if rel >= 1 else "/"
        note = f"{rel:6.1f}x" if rel >= 1 else f"{1/rel:6.1f}x slower"
        print(f"  {name:22} {secs:8.3f} s   {note}")
    print()
