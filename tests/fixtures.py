"""Known coordinates for testing CRS detection.

Each entry is (name, x, y, expected_crs).

PROVENANCE - read this before trusting a number:

  EPSG:4326  (WGS84 lon/lat)  - well-known city coordinates.
  EPSG:3857  (Web Mercator)   - COMPUTED from the 4326 values with the exact
                                spherical formula. These are reliable.
  EPSG:28992 (Dutch RD New)   - APPROXIMATE. Verify on https://epsg.io/28992
  EPSG:27700 (British Nat Grid) - APPROXIMATE. Verify on https://epsg.io/27700

The 28992 and 27700 transforms need datum shifts (and OSTN for the UK), so the
values below are good enough to land inside the right *range* - which is all a
range-based detector needs - but are NOT survey-accurate. Do not use them to
test a reprojection function.
"""

# --- EPSG:4326 - WGS84 longitude/latitude -----------------------------------
# Tell: lon in [-180, 180], lat in [-90, 90], usually with decimals.
WGS84 = [
    ("Amsterdam", 4.9041, 52.3676, "EPSG:4326"),
    ("London", -0.1278, 51.5074, "EPSG:4326"),
    ("New York", -74.0060, 40.7128, "EPSG:4326"),
    ("Sydney", 151.2093, -33.8688, "EPSG:4326"),
]

# --- EPSG:3857 - Web Mercator (computed, exact) -----------------------------
# Tell: magnitudes in the millions; bounded by +/- 20037508.34
WEB_MERCATOR = [
    ("Amsterdam", 545921.9, 6866867.1, "EPSG:3857"),
    ("London", -14226.6, 6711542.5, "EPSG:3857"),
    ("New York", -8238310.2, 4970071.6, "EPSG:3857"),
    ("Sydney", 16832542.3, -4011198.6, "EPSG:3857"),
]

# --- EPSG:28992 - Dutch RD New (APPROXIMATE) --------------------------------
# Tell: X roughly 0-300k, Y roughly 300k-630k.
# Origin is Amersfoort at (155000, 463000) by definition.
RD_NEW = [
    ("Amersfoort origin", 155000.0, 463000.0, "EPSG:28992"),
    ("Amsterdam", 121000.0, 487000.0, "EPSG:28992"),
    ("Rotterdam", 92000.0, 437000.0, "EPSG:28992"),
    ("Groningen", 233000.0, 582000.0, "EPSG:28992"),
]

# --- EPSG:27700 - British National Grid (APPROXIMATE) -----------------------
# Tell: E 0-700k, N 0-1300k.
BNG = [
    ("London", 530000.0, 180000.0, "EPSG:27700"),
    ("Edinburgh", 325000.0, 673000.0, "EPSG:27700"),
    ("Cardiff", 318000.0, 176000.0, "EPSG:27700"),
]

ALL = WGS84 + WEB_MERCATOR + RD_NEW + BNG


# --- AMBIGUOUS - these SHOULD NOT resolve to a single answer -----------------
# RD New and British National Grid overlap in numeric range. A detector that
# confidently picks one of these is lying. Good behaviour: report both, or
# report the column-level verdict from the majority of rows.
AMBIGUOUS = [
    (300000.0, 500000.0),  # valid in both 28992 and 27700
    (200000.0, 600000.0),  # ditto
]

# --- NONSENSE - should come back "unknown", not a guess ----------------------
UNKNOWN = [
    (1e12, 1e12),
    (float("nan"), float("nan")),
]

# --- SENTINELS - a known limitation, not a bug ------------------------------
# These are conventional "missing data" markers, but they fall inside Web
# Mercator's range, so a range-based detector reports EPSG:3857. It is not
# wrong -- (-999, -999) really is a valid Web Mercator point about 1 km from
# Null Island. There is no way to tell the two apart from the number alone.
# Callers should strip sentinels before detection.
SENTINELS_LOOK_LIKE_3857 = [
    (-999.0, -999.0),
    (-9999.0, -9999.0),
    (0.0, 0.0),  # also valid WGS84 (Null Island)
]
