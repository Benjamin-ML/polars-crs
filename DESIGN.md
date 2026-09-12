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

## Performance

Nothing in the per-row path allocates. Matches are a `u8` bitmask rather than a
`Vec`, labels are borrowed `&'static str` rather than owned `String`, and the
sixteen possible candidate strings are rendered once on first use rather than
formatted per row.

The remaining cost is building the output `String` column, which dominates the
runtime. An integer or categorical output would be materially faster.

## Out of scope

- **Reprojection.** Datum shifts and OSTN grids are pyproj's job.
- **Geometry operations.** Use polars-st.
- **UTM zone detection.** Overlaps RD New and BNG heavily.
