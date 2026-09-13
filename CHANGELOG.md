# Changelog

## 0.1.7

- **The aggregating functions run in parallel.** `detect`, `detect_candidates`
  and `detect_report` were still single-threaded; only the elementwise
  functions had been parallelised. On 10,000,000 rows `detect` drops from
  95 ms to 29 ms.

  The per-row checks added in 0.1.2 and 0.1.3 had cost about 23%, taking
  `detect` from 77 ms to 95 ms. This more than recovers it. The tallies are
  counts, so slices merge by addition and the result does not depend on how
  the work was divided.

## 0.1.6

Fixes three regressions introduced by the 0.1.5 input guard.

- **Decimal columns aborted the process.** `polars` was built without
  `dtype-decimal`, so the cast panicked inside a non-unwinding boundary and
  raised SIGABRT, which no caller could catch. The dtype features are now
  enabled, and Decimal works rather than being rejected.
- **`group_by().agg()` and `over()` were rejected.** The duplicate-axis check
  compared column names, and Polars hands a plugin unnamed series inside an
  aggregation, so both arrived empty and looked identical. The check now
  compares the values, stopping at the first difference.
- **`Int8`, `Int16`, `UInt8` and `UInt16` panicked.** Same missing dtype
  features. All integer widths now work.

The name-based check was also wrong in both directions: it rejected two
different columns sharing an alias, rejected two literals, and accepted the same
column under two names. It now compares values, exempting constant columns,
since sentinel data such as a column of `(0, 0)` or `(-999, -999)` legitimately
holds the same value on both axes.

## 0.1.5

- Rejects Booleans and temporal types. `True`/`False` cast to 1.0/0.0 and dates
  cast to their epoch day, all inside the lon/lat box, so they detected as a
  confident `EPSG:4326`.
- Rejects the same column passed as both axes.
- `detect_report` carries an absolute row count as a third field,
  `CODE=narrowest/contains/rows`, so trace contamination survives rounding.
- A column that fits WGS84 both ways round is marked `axis-order=unverifiable`.

**Behaviour change:** numeric strings such as `"155000"` were accepted in 0.1.4
and are now rejected. Cast explicitly with `pl.col("x").cast(pl.Float64)`.

## 0.1.4

- The verdict is decided only on exclusive matches. Containment could
  previously make a range win a column it had no coordinates in, both for
  mixtures containing Web Mercator and inside the band between the mixture
  floor and the single-label threshold.
- The Null Island guard uses an epsilon rather than an exact comparison.
- The dropped sentinel count is reported against the row total.

## 0.1.3

- Detects swapped x/y axes where a longitude beyond +/-90 gives it away.
- Drops Null Island sentinels before scoring, so a column of zeros returns
  `unknown` rather than a confident `EPSG:4326`.

## 0.1.2

- A column holding coordinates from two systems no longer reports a third.
  Returns `mixed:EPSG:4326|EPSG:28992` instead.
- `detect_report` reads `CODE=narrowest/contains`.

## 0.1.1

- Declares `polars` as a runtime dependency. 0.1.0 failed on first import in a
  clean environment.
- Adds description, README, licence, project URLs and classifiers.

## 0.1.0

First release.
