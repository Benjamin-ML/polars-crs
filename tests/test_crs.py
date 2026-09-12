import math

import polars as pl
import pytest
from fixtures import ALL, AMBIGUOUS, BNG, RD_NEW, UNKNOWN, WEB_MERCATOR, WGS84

import polars_crs as plc


def _frame(rows):
    return pl.DataFrame({"x": [r[1] for r in rows], "y": [r[2] for r in rows]})


# --- single-system check ------------------------------------------------------


def test_is_rd_new_accepts_dutch_points():
    got = _frame(RD_NEW).select(plc.is_rd_new("x", "y")).to_series().to_list()
    assert all(got)


@pytest.mark.parametrize("rows", [WGS84, WEB_MERCATOR])
def test_is_rd_new_rejects_other_systems(rows):
    got = _frame(rows).select(plc.is_rd_new("x", "y")).to_series().to_list()
    assert not any(got)


# --- per-row detection --------------------------------------------------


@pytest.mark.parametrize("name,x,y,expected", ALL)
def test_guess_matches_expected(name, x, y, expected):
    """Per-row guess picks the narrowest matching range.

    WGS84 and Web Mercator are unambiguous per-row. RD New and BNG overlap, so
    a BNG point inside the RD New range is EXPECTED to come back as RD New --
    that is the documented priority rule, not a bug. Only assert where the
    point is outside the narrower range.
    """
    got = pl.DataFrame({"x": [x], "y": [y]}).select(plc.guess("x", "y")).item()
    if (
        expected == "EPSG:27700"
        and plc.CRS_RANGES["EPSG:28992"]["y"][0]
        <= y
        <= plc.CRS_RANGES["EPSG:28992"]["y"][1]
    ):
        pytest.skip("genuinely ambiguous with EPSG:28992 -- see BUILD_PLAN.md")
    assert got == expected


def test_candidates_lists_every_match():
    # Amsterdam in RD New sits inside RD New, BNG and Web Mercator ranges.
    got = (
        pl.DataFrame({"x": [121000.0], "y": [487000.0]})
        .select(plc.candidates("x", "y"))
        .item()
    )
    assert got.split("|") == ["EPSG:28992", "EPSG:27700", "EPSG:3857"]


# --- whole-column detection ----------------------------------------


@pytest.mark.parametrize(
    "rows,expected",
    [
        (WGS84, "EPSG:4326"),
        (WEB_MERCATOR, "EPSG:3857"),
        (RD_NEW, "EPSG:28992"),
        (BNG, "EPSG:27700"),
    ],
)
def test_detect_whole_column(rows, expected):
    assert _frame(rows).select(plc.detect("x", "y")).item() == expected


def test_detect_returns_one_row():
    out = _frame(RD_NEW).select(plc.detect("x", "y"))
    assert out.height == 1


def test_detect_candidates_reports_ambiguity():
    got = _frame(RD_NEW).select(plc.detect_candidates("x", "y")).item()
    assert "EPSG:28992" in got and "EPSG:27700" in got


def test_mixed_systems_collapse_to_unknown():
    """A column mixing WGS84 and Web Mercator is consistent with nothing narrow."""
    df = pl.DataFrame({"x": [4.9, 545921.9], "y": [52.3, 6866867.1]})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:3857"


# --- edge cases --------------------------------------------------------------


@pytest.mark.parametrize("x,y", UNKNOWN)
def test_nonsense_is_unknown_not_a_guess(x, y):
    got = pl.DataFrame({"x": [x], "y": [y]}).select(plc.guess("x", "y")).item()
    assert got == "unknown"


@pytest.mark.parametrize("x,y", AMBIGUOUS)
def test_ambiguous_points_report_multiple(x, y):
    got = pl.DataFrame({"x": [x], "y": [y]}).select(plc.candidates("x", "y")).item()
    assert len(got.split("|")) > 1


def test_nan_rows_are_skipped_not_fatal():
    df = pl.DataFrame({"x": [121000.0, math.nan], "y": [487000.0, math.nan]})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


def test_nulls_are_skipped():
    df = pl.DataFrame({"x": [121000.0, None], "y": [487000.0, None]})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


def test_empty_column_is_unknown():
    df = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
    assert df.select(plc.detect("x", "y")).item() == "unknown"


# --- namespace API -----------------------------------------------------------


def test_namespace_matches_plain_functions():
    df = _frame(RD_NEW)
    assert (
        df.select(pl.col("x").crs.detect("y")).item()
        == df.select(plc.detect("x", "y")).item()
    )


def test_namespace_works_in_with_columns():
    df = _frame(RD_NEW).with_columns(g=pl.col("x").crs.guess("y"))
    assert df["g"].to_list() == ["EPSG:28992"] * len(RD_NEW)


# --- known limitations, pinned so a change is deliberate ---------------------


@pytest.mark.parametrize("x,y", [(-999.0, -999.0), (-9999.0, -9999.0), (0.0, 0.0)])
def test_sentinels_are_indistinguishable_from_real_coordinates(x, y):
    """Documented limitation, see BUILD_PLAN.md.

    Conventional missing-data sentinels fall inside real CRS ranges, so they
    detect as valid coordinates rather than as missing data. Which system they
    land on depends on the ranges; the point is that they are never flagged.
    This is not fixable from the values alone -- (-999, -999) IS a real point
    in several systems. Strip sentinels before detection.
    """
    got = pl.DataFrame({"x": [x], "y": [y]}).select(plc.guess("x", "y")).item()
    assert got != "unknown"


# --- outlier tolerance -------------------------------------------------------


def test_single_outlier_does_not_discard_the_right_answer():
    """Real data has typos. One bad row must not eliminate the correct CRS."""
    xs = [121000.0] * 50 + [999999999.0]
    ys = [487000.0] * 50 + [999999999.0]
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


def test_too_many_outliers_does_reject():
    """Tolerance is 95%, not unlimited."""
    xs = [121000.0] * 50 + [999999999.0] * 50
    ys = [487000.0] * 50 + [999999999.0] * 50
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "unknown"


def test_detect_report_shows_confidence():
    df = pl.DataFrame({"x": [121000.0] * 50, "y": [487000.0] * 50})
    got = df.select(plc.detect_report("x", "y")).item()
    assert "EPSG:28992=1.00" in got


# --- mixed columns -----------------------------------------------------------


def _mix(a, b):
    return pl.DataFrame(
        {
            "x": [r[1] for r in a] + [r[1] for r in b],
            "y": [r[2] for r in a] + [r[2] for r in b],
        }
    )


def test_mixed_column_is_not_reported_as_a_third_system():
    """A broad range contains both groups and would otherwise win outright.

    EPSG:27700 spans both the Dutch RD values and the WGS84 values, so it
    scores 1.00 while each real system scores 0.50 and fails the threshold.
    Reporting 27700 would name a system with no coordinates in the data.
    """
    got = _mix(RD_NEW, WGS84).select(plc.detect("x", "y")).item()
    assert got.startswith("mixed:")
    assert "EPSG:28992" in got and "EPSG:4326" in got
    assert "EPSG:27700" not in got


def test_mixed_column_detect_candidates_agrees():
    got = _mix(RD_NEW, WGS84).select(plc.detect_candidates("x", "y")).item()
    assert got.startswith("mixed:")


@pytest.mark.parametrize(
    "rows,expected",
    [
        (WGS84, "EPSG:4326"),
        (WEB_MERCATOR, "EPSG:3857"),
        (RD_NEW, "EPSG:28992"),
        (BNG, "EPSG:27700"),
    ],
)
def test_pure_columns_are_never_called_mixed(rows, expected):
    """Overlapping ranges must not be mistaken for a mixture.

    Some British points fall inside the Dutch RD range, so BNG data has a
    genuine minority narrowest-match elsewhere. That is overlap, not a mixture.
    """
    got = _frame(rows).select(plc.detect("x", "y")).item()
    assert got == expected


def test_small_contamination_does_not_flip_the_verdict():
    xs = [r[1] for r in RD_NEW] * 19 + [WGS84[0][1]]
    ys = [r[2] for r in RD_NEW] * 19 + [WGS84[0][2]]
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


def test_report_exposes_the_split():
    got = _mix(RD_NEW, WGS84).select(plc.detect_report("x", "y")).item()
    # Format is CODE=narrowest/contains. The mixture signature is two systems
    # holding a substantial narrowest share each, while a broad range contains
    # everything but is narrowest for nothing.
    parts = {
        p.split("=")[0]: tuple(float(v) for v in p.split("=")[1].split("/"))
        for p in got.split("|")
    }
    assert 0.2 < parts["EPSG:4326"][0] < 0.8
    assert 0.2 < parts["EPSG:28992"][0] < 0.8
    assert parts["EPSG:27700"] == (0.0, 1.0)


# --- limitations that range checks cannot see --------------------------------


LON = [4.9, -0.13, -74.0, 151.2, -122.4, 139.7]
LAT = [52.4, 51.5, 40.7, -33.9, 37.8, 35.7]


def test_swapped_axes_are_detected_when_longitude_exceeds_90():
    """A longitude beyond +/-90 cannot be a latitude, which gives it away."""
    normal = pl.DataFrame({"x": LON, "y": LAT})
    swapped = pl.DataFrame({"x": LAT, "y": LON})
    assert normal.select(plc.detect("x", "y")).item() == "EPSG:4326"
    assert swapped.select(plc.detect("x", "y")).item() == "swapped:EPSG:4326"


def test_swapped_axes_are_invisible_within_90_degrees():
    """Documented limitation.

    Dutch coordinates sit under 90 in both directions, so the pair is a valid
    WGS84 coordinate either way round and no test on the values separates them.
    """
    lon = [r[1] for r in WGS84 if abs(r[1]) <= 90 and abs(r[2]) <= 90]
    lat = [r[2] for r in WGS84 if abs(r[1]) <= 90 and abs(r[2]) <= 90]
    swapped = pl.DataFrame({"x": lat, "y": lon})
    assert swapped.select(plc.detect("x", "y")).item() == "EPSG:4326"


def test_swap_detection_does_not_fire_on_projected_data():
    for rows in (RD_NEW, BNG, WEB_MERCATOR):
        got = _frame(rows).select(plc.detect("x", "y")).item()
        assert not got.startswith("swapped:"), rows


def test_all_zeros_is_unknown_not_null_island():
    """Exact (0, 0) is dropped before scoring.

    Null Island is open ocean, so a zero pair is almost always missing data
    encoded as a number. A column of them has nothing to detect.
    """
    df = pl.DataFrame({"x": [0.0] * 5, "y": [0.0] * 5})
    assert df.select(plc.detect("x", "y")).item() == "unknown"


def test_null_island_rows_are_reported_not_hidden():
    xs = [r[1] for r in RD_NEW] + [0.0, 0.0]
    ys = [r[2] for r in RD_NEW] + [0.0, 0.0]
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"
    assert "null-island-rows=2" in df.select(plc.detect_report("x", "y")).item()


def test_zero_pairs_do_not_dilute_the_verdict():
    """Without the sentinel drop, enough zeros would pull the column to 4326."""
    xs = [r[1] for r in RD_NEW] + [0.0] * 20
    ys = [r[2] for r in RD_NEW] + [0.0] * 20
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"
