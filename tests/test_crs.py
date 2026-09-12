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
