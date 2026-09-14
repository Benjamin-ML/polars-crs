# Changelog

## 0.1.8

Three bugs of the same shape: rows that fit no narrow range fell into British
National Grid, whose box is wide enough to catch almost anything.

- **Partial swaps produced a phantom.** Two sources merged with opposite axis
  order is the realistic case. Between 50 and 90 percent reversed the column was
  labelled part British, and below 50 percent the reversed rows were attributed
  to BNG in the report. Axis order is now decided per row rather than per frame,
  so a partial swap reads as `mixed:EPSG:4326|swapped:EPSG:4326`.
- **Transposed Dutch RD was labelled EPSG:27700 at full confidence.** Turning RD
  round moves it out of the Dutch box and into the British one. This is not
  decidable from the values, since genuine southern England occupies the same
  region, so it is now raised in `detect_report` as
  `transposed-candidate:EPSG:28992` rather than acted on.
- **Only `(0, 0)` was treated as a placeholder.** `(-999, -999)`,
  `(-9999, -9999)` and the rest are at least as common in exported data, and a
  file of them read as British with full confidence. Any repeated point on the
  diagonal holding 5 percent or more of the rows is now dropped and reported as
  `placeholder[value]=N/total`.

Also:

- The verdict and the per-system breakdown no longer disagree; both are built
  from the same readings.
- `Float64` input skips a redundant cast.
- `numpy` added to the test requirements.

Rows whose axis order cannot be checked, where the pair reads identically either
way round, are no longer counted as evidence for the given order. They are
apportioned by the rows that can be told apart, so a fully reversed column reads
as reversed rather than as a mixture.

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
