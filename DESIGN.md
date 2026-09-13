# Design

## Problem

Coordinate data arrives as two numeric columns with no metadata. The
projection is not recorded, and the existing tooling requires you to already
know it: GeoPandas and pyproj make you declare a CRS, and polars-st reads an
SRID that must already be present in the geometry.

Getting it wrong fails silently. Treating projected metres as degrees produces
a plausible number, not an error.

## Approach

Each coordinate reference system occupies a distinctive numeric range. A
coordinate outside a system's range cannot belong to that system, so detection
is elimination rather than classification.

| Code | System | X range | Y range |
|---|---|---|---|
| EPSG:4326 | WGS84 lon/lat | -180 ... 180 | -90 ... 90 |
| EPSG:28992 | Dutch RD New | -1,000 ... 290,000 | 300,000 ... 640,000 |
| EPSG:27700 | British National Grid | -110,000 ... 690,000 | -20,000 ... 1,260,000 |
| EPSG:3857 | Web Mercator | +/-20,037,508 | +/-20,048,966 |

Bounds come from PROJ: each system's official area of use, sampled on a
200x200 grid, transformed into projected coordinates, then padded. They are not
hand-written, because a range that is slightly too narrow fails silently.

`validate.py` regenerates the check against pyproj ground truth.

## Ordering

`CRS_DEFS` is ordered narrowest range first. When several systems match, the
narrowest is the most specific answer, so `guess` and `detect` both take the
first survivor. The ordering is load-bearing.

Every WGS84 coordinate is also numerically valid Web Mercator, since 4326's
range sits inside 3857's. Treating that as ambiguous would make the tool
useless, so 4326 wins: a Web Mercator coordinate that small would be within
about 180 m of Null Island, which is open ocean. This is a deliberate
heuristic and is documented rather than hidden.

## Tolerance

`detect` keeps every system containing at least 95% of the rows. Real data has
typos, sentinels and points just outside an official area of use; requiring
every row to match would let one bad row discard the correct answer.

`detect_report` exposes the match fractions so callers can see how firm the
verdict is.

## The decision rule

Only the narrowest-match shares decide the verdict. They are exclusive and sum
to one, so they are the only evidence that cannot be inflated by a wide range.

Containment is diagnostic. Letting it decide produced a confident wrong answer
twice: first when a broad range contained both halves of a mixed column, and
again in the band where no system reached the single-label threshold and the
verdict fell through to whatever contained the most rows. Both times the answer
named a system with no coordinates in the data.

## Mixtures

Counting only how many rows each range contains is not enough. A broad range
contains every group in a mixed column and scores 1.00, while each real system
scores its own share and falls below the threshold, so the broad one wins and
names a system that is not present.

Each row's *narrowest* match is therefore tracked separately. Those shares do
sum to one, and a mixture shows up as two or more narrower systems each holding
a meaningful share while a broad one is narrowest for nothing.

Overlap is not a mixture. Some British points fall inside the Dutch range, so
pure EPSG:27700 data has a genuine minority narrowest elsewhere. Requiring two
components above `MIXTURE_MIN` that together reach the threshold separates the
two cases.

## Performance

Nothing in the per-row path allocates. Matches are a `u8` bitmask rather than a
`Vec`, labels are borrowed `&'static str` rather than owned `String`, and the
sixteen possible candidate strings are rendered once on first use rather than
formatted per row.

Elementwise work is split across cores in `src/parallel.rs`, and the
aggregating functions split the same way in `src/detect.rs`. Their accumulators
are counts, so slices merge by addition and the result is independent of how the
work was divided. Below 100,000 rows the split is skipped, since thread setup
costs more than it saves.

## Axis order and sentinels

Two defects are common enough in real files to be worth handling rather than
only documenting.

**Swapped axes.** Detected by testing whether the column would fit WGS84 with x
and y exchanged while it does not fit as given. That only separates the two
where some row carries a longitude beyond +/-90, since anything inside the
latitude range is a valid coordinate either way round.

**Null Island.** Anything within 1e-9 of (0, 0) is dropped before scoring. An
epsilon rather than an exact test, since float arithmetic produces denormals and
near-zero residue where zero was meant. It is open ocean, so it
is almost always a missing value encoded as a number, and counting it lets a
column of missing data read as confident WGS84. Dropped rows are counted and
reported rather than silently discarded.

## Out of scope

- **Reprojection.** Datum shifts and OSTN grids are pyproj's job.
- **Geometry operations.** Use polars-st.
- **UTM zone detection.** Overlaps RD New and BNG heavily.
