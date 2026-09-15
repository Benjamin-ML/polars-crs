# polars-crs

Detect which coordinate reference system a column of unlabelled x/y numbers is
in, by looking at the values.

```python
import polars as pl
import polars_crs as plc

df = pl.DataFrame({"x": [121000.0, 92000.0], "y": [487000.0, 437000.0]})

df.select(plc.detect("x", "y")).item()
# 'EPSG:28992'   (Dutch RD New)
```

## Why you would want this

You get a CSV with two numeric columns and no metadata. Nobody recorded the
projection. Every existing tool requires you to already know it: GeoPandas and
pyproj make you declare a CRS, and polars-st reads an SRID that has to be there
already.

Guessing wrong fails silently. Treat projected metres as degrees and the
pipeline runs, the plot renders, and the number is wrong:

```
Amsterdam to Rotterdam, assuming lat/lon:   6,612.5 km
Amsterdam to Rotterdam, after detecting:       57.7 km
```

No exception, no warning. `detect` turns that into a value you can assert on.

## What you actually get

A range check is easy to write yourself. Twenty lines of `when/then` will label
most rows correctly, and on a clean column it will agree with this package. The
difference is what happens on the columns that are not clean, which is most of
the ones worth checking.

| | |
|---|---|
| **Bounds derived from PROJ** | Each system's official area of use, sampled and transformed, not numbers typed from memory. A range slightly too narrow fails silently: the rows outside it simply eliminate the correct answer |
| **Validated against ground truth** | `validate.py` generates real coordinates with pyproj and asks the detector to identify them |
| **Mixed columns are named, not guessed** | Two sources merged returns `mixed:EPSG:4326\|EPSG:28992` rather than a third system that happens to contain both |
| **Reversed axes are caught** | `lat,lon` instead of `lon,lat` returns `swapped:EPSG:4326`, including when only part of the file is reversed |
| **Placeholders are not coordinates** | `(0,0)`, `(-999,-999)` and friends are real points somewhere; repeated, they are missing data, and are dropped and counted |
| **Overlap is not mistaken for ambiguity** | British points genuinely fall inside the Dutch range. That is reported as overlap, not as a mixed column |
| **It says what it could not check** | `axis-order=unverifiable` when both orders are valid, `transposed-candidate` when a transposition is plausible but undecidable |
| **Trace contamination stays visible** | One bad row in 100,000 rounds to `0.00` but still reports as `/1` |

Every one of those exists because it was found failing. The package is the
accumulated result of nine rounds of adversarial testing, not a first draft.

**On speed:** `detect` is about 3.6x faster than the nearest native equivalent
on 10M rows. The elementwise `guess` is at parity with a hand-written
`when/then` chain, so speed is not the reason to use it. Correctness is. See
[Performance](#performance).

**What it is:** a triage tool for unlabelled files. Not an authority. Confirm a
verdict against a known landmark before reprojecting anything.

## Install

```bash
pip install polars-crs
```

Requires `polars >= 1.4.0`. Earlier versions cannot run a scalar-returning
plugin inside `group_by`, so the aggregating functions fail there.

Wheels are published for Linux (x86_64, aarch64), Windows (x64) and macOS
(Apple Silicon). Intel macOS builds from the sdist and needs a Rust toolchain.

## API

| Function | Shape | Returns |
|---|---|---|
| `detect(x, y)` | aggregation | One verdict for the whole column |
| `detect_candidates(x, y)` | aggregation | Every system still consistent with all rows |
| `guess(x, y)` | elementwise | Narrowest matching system, per row |
| `candidates(x, y)` | elementwise | Every system whose range contains the row, pipe-joined. Raw containment, deliberately: it exists to show overlap |
| `detect_report(x, y)` | aggregation | Per-system detail, see [Reading the report](#reading-the-report) |
| `is_rd_new(x, y)` | elementwise | Boolean, EPSG:28992 only |

All are also available as an expression namespace: `pl.col("x").crs.detect("y")`.

## How it works

Each system occupies a distinctive numeric range. A coordinate outside a
system's range cannot be in that system.

| Code | System | X range | Y range |
|---|---|---|---|
| EPSG:4326 | WGS84 lon/lat | −180 … 180 | −90 … 90 |
| EPSG:28992 | Dutch RD New | −1,000 … 290,000 | 300,000 … 640,000 |
| EPSG:27700 | British National Grid | −110,000 … 690,000 | −20,000 … 1,260,000 |
| EPSG:3857 | Web Mercator | ±20,037,508 | ±20,048,966 |

Bounds are **derived from PROJ**, not hand-written: each system's official area
of use, sampled on a 200×200 grid and transformed into projected coordinates.

`detect` keeps every system containing at least **95%** of the rows and returns
the narrowest. The threshold matters - real data has typos, sentinels and
points just outside an official area of use, and requiring every row to match
made a single outlier discard the correct answer.

## Validation

`validate.py` checks the detector against PROJ ground truth: it generates real
coordinates in each system with pyproj, then asks the detector to identify
them.

```
EPSG:28992   column verdict EPSG:28992    per-row exact 100.0%
EPSG:27700   column verdict EPSG:27700    per-row exact  87.2%
EPSG:3857    column verdict EPSG:3857     per-row exact  99.8%
EPSG:4326    column verdict EPSG:4326     per-row exact 100.0%

Whole-column detection: 4/4 correct
```

Per-row accuracy for EPSG:27700 is 87.2% because British points that also fall
inside the Dutch RD New range are labelled as the narrower system. That is the
documented priority rule, and why `candidates` exists.

Bounds are checked against PROJ rather than assumed, because a range that is
slightly too narrow fails silently: the coordinates outside it simply eliminate
the correct system.

## Limitations - read these

1. **Range-based.** It narrows to candidates; it never proves.
2. **RD New and British National Grid genuinely overlap.** `(300000, 500000)`
   is valid in both. `guess` and `detect` take the narrowest range;
   `candidates` and `detect_candidates` show you everything.
3. **4326 is preferred over 3857 when both match.** Every lat/lon pair is also
   numerically valid Web Mercator, so this is a deliberate heuristic - a 3857
   coordinate that small would be within ~180 m of Null Island, open ocean.
4. **Missing-data sentinels are never flagged.** `-999`, `-9999` and `(0, 0)`
   all fall inside real CRS ranges, so they detect as valid coordinates rather
   than as missing data. Not fixable from the values alone: `(-999, -999)`
   genuinely *is* a real point in several systems. Strip sentinels first.
5. **Bounds come from each system's official area of use.** A coordinate
   legitimately outside that area - an extrapolated projection - will not be
   recognised.
6. **No reprojection.** Converting between systems needs datum shifts and OSTN
   grids - that is pyproj's job and it does it properly.
7. **Swapped axes are only caught outside +/-90.** A reversed pair is detected
   when some row carries a longitude beyond the latitude range. Within it, both
   orders are valid WGS84 coordinates and the values cannot distinguish them.
8. **Wheels do not cover every platform.** Intel macOS and musl-based images
   such as Alpine fall back to building the sdist, which needs a Rust toolchain.

This is a triage tool for unlabelled files, not an authority. Confirm a verdict
against a known landmark before reprojecting anything.

## Performance

Measured on 0.1.8, best of five, release build, Apple Silicon with 10 cores.

The honest comparison is against native Polars expressions, since a range check
is expressible as a `when/then` chain without any Rust:

| rows | `guess` | native `when/then` | `detect` | nearest native equivalent |
|---|---|---|---|---|
| 100,000 | 0.57 ms | 0.51 ms | 0.44 ms | 0.97 ms |
| 1,000,000 | 1.81 ms | 2.54 ms | **2.00 ms** | 9.50 ms |
| 10,000,000 | 23.46 ms | 23.38 ms | **27.55 ms** | 99.04 ms |

`guess` is at parity. There is no speed argument for it.

`detect` is 3.6x faster at 10M rows, because it makes one pass accumulating
counts across cores rather than materialising a label column and aggregating it.

The "nearest native equivalent" is not actually equivalent: it computes the
modal narrowest label and nothing else. It does not detect reversed axes, does
not distinguish a mixture from range overlap, does not recognise placeholders,
and does not report what it could not check. Those are the reasons to use this,
and none of them is practical to write as an expression chain.

### For scale

The alternatives people reach for first, on 1,000,000 rows:

| Implementation | Time |
|---|---|
| plugin / native `when/then` | ~2 ms |
| numpy | 83 ms |
| pure Python loop | 241 ms |
| `map_elements` | 418 ms |

`map_elements` is the one worth avoiding: it calls back into Python once per
row. But benchmarking a plugin only against it is measuring against a straw man,
which is why the table above uses native expressions instead.

Both the elementwise and the aggregating functions are split across cores, so
throughput scales with the machine rather than with a single thread.

## Development

```bash
source ../.venv/bin/activate
export PATH="$HOME/.cargo/bin:$PATH"

cargo check            # fast typecheck while writing
cargo clippy           # lint
maturin develop        # debug build
maturin develop --release   # optimized - use this before benchmarking
pytest tests/        # 42 tests
python bench_full.py # five implementations compared
python gendata.py N  # generate N rows of realistic test data
python validate.py   # accuracy vs pyproj ground truth
python demo_why.py   # why getting this wrong is expensive
```

`pyproj` and `numpy` are needed for validation and benchmarking only, not at
runtime.

Note `maturin develop` builds **unoptimized**. Never quote a benchmark from it.
