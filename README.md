# polars-crs

Detect which coordinate reference system a column of unlabelled `x`/`y` numbers
is in, by looking at the values.

You get a CSV with two numeric columns and no metadata. Nobody tells you the
projection. Every existing tool requires you to already know - GeoPandas and
pyproj make you declare it, polars-st reads an SRID that must already be there.

```python
import polars as pl
import polars_crs as plc

df = pl.DataFrame({"x": [121000.0, 92000.0], "y": [487000.0, 437000.0]})

df.select(plc.detect("x", "y")).item()
# 'EPSG:28992'   (Dutch RD New)

df.with_columns(pl.col("x").crs.guess("y"))
# per-row labels
```

## API

| Function | Shape | Returns |
|---|---|---|
| `detect(x, y)` | aggregation | One verdict for the whole column |
| `detect_candidates(x, y)` | aggregation | Every system still consistent with all rows |
| `guess(x, y)` | elementwise | Narrowest matching system, per row |
| `candidates(x, y)` | elementwise | All matching systems, pipe-joined, per row |
| `detect_report(x, y)` | aggregation | Match fraction per system, e.g. `EPSG:28992=1.00\|EPSG:3857=1.00` |
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

## Performance

Five implementations of `guess`, identical output, best of three, release
build, Apple Silicon with 10 cores:

| Implementation | 1,000,000 rows | 10,000,000 rows |
|---|---|---|
| **plugin (Rust)** | **0.003 s** | **0.020 s** |
| polars native `when/then` | 0.005 s | 0.042 s |
| numpy | 0.083 s | 0.873 s |
| `map_elements` | 0.418 s | (too slow) |
| pure Python loop | 0.241 s | (too slow) |

Elementwise work is split across cores, so throughput scales with the machine
rather than with a single thread.

Beyond speed:

- **Correct bounds**, derived from PROJ areas of use and validated against
  ground truth. This is the part that is actually hard, and that a hand-rolled
  `when/then` chain with guessed numbers gets wrong.
- **The aggregating functions.** `detect` and `detect_report` eliminate
  candidates across rows with a tolerance threshold, which is awkward to
  express as an expression chain.
- **An API.** `pl.col("x").crs.detect("y")` rather than every user writing and
  maintaining twenty lines of `when/then` with the bounds inlined.

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
