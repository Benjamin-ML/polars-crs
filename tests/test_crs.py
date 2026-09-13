import datetime
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


def test_detect_candidates_lists_narrowest_matches_only():
    """Containment is diagnostic, not a candidate.

    EPSG:27700 contains every Dutch point but is the narrowest match for none,
    so it is not a candidate. detect_report still shows the containment.
    """
    got = _frame(RD_NEW).select(plc.detect_candidates("x", "y")).item()
    assert got == "EPSG:28992"
    assert (
        "EPSG:27700=0.00/1.00"
        in _frame(RD_NEW).select(plc.detect_report("x", "y")).item()
    )


def test_mixture_containing_web_mercator_is_flagged():
    """Web Mercator contains almost everything, so it used to win on
    containment and hide the other half of the column."""
    df = pl.DataFrame({"x": [4.9, 545921.9], "y": [52.3, 6866867.1]})
    got = df.select(plc.detect("x", "y")).item()
    assert got.startswith("mixed:")
    assert "EPSG:4326" in got and "EPSG:3857" in got


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
    assert set(got.split("|")) == {"EPSG:4326", "EPSG:28992"}


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
        if "=" in p and "/" in p
    }
    # CODE=narrowest/contains/rows
    assert 0.2 < parts["EPSG:4326"][0] < 0.8
    assert 0.2 < parts["EPSG:28992"][0] < 0.8
    assert parts["EPSG:27700"][:2] == (0.0, 1.0)
    assert parts["EPSG:27700"][2] == 0.0


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
    assert "null-island=2/" in df.select(plc.detect_report("x", "y")).item()


def test_zero_pairs_do_not_dilute_the_verdict():
    """Without the sentinel drop, enough zeros would pull the column to 4326."""
    xs = [r[1] for r in RD_NEW] + [0.0] * 20
    ys = [r[2] for r in RD_NEW] + [0.0] * 20
    df = pl.DataFrame({"x": xs, "y": ys})
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


# --- containment must never decide the verdict -------------------------------

RD_PT = (121000.0, 487000.0)
WGS_PT = (4.9041, 52.3676)
MERC_PT = (545921.9, 6866867.1)


def _pts(pairs):
    return pl.DataFrame({"x": [p[0] for p in pairs], "y": [p[1] for p in pairs]})


@pytest.mark.parametrize("dominant", [0.80, 0.90, 0.905, 0.94, 0.95, 0.99, 1.0])
def test_no_phantom_system_at_any_contamination_level(dominant):
    """There must be no band where a merely-containing range wins.

    Between the mixture floor and the single-label threshold, nothing used to
    qualify and the verdict fell through to EPSG:27700, which contains both
    Dutch and WGS84 values while being preferred by neither.
    """
    n = 1000
    k = round(dominant * n)
    got = _pts([RD_PT] * k + [WGS_PT] * (n - k)).select(plc.detect("x", "y")).item()
    assert got == "EPSG:28992", f"{dominant}: {got}"


@pytest.mark.parametrize(
    "pairs,expected",
    [
        ([(RD_PT, 50), (MERC_PT, 50)], {"EPSG:28992", "EPSG:3857"}),
        (
            [(RD_PT, 33), (WGS_PT, 33), (MERC_PT, 33)],
            {"EPSG:4326", "EPSG:28992", "EPSG:3857"},
        ),
        ([(WGS_PT, 50), (MERC_PT, 50)], {"EPSG:4326", "EPSG:3857"}),
    ],
)
def test_mixtures_with_web_mercator_are_not_reported_as_mercator(pairs, expected):
    """EPSG:3857 contains nearly everything, so containment made it swallow
    any column it was merged into."""
    rows = [p for pt, n in pairs for p in [pt] * n]
    got = _pts(rows).select(plc.detect("x", "y")).item()
    assert got.startswith("mixed:")
    assert set(got.removeprefix("mixed:").split("|")) == expected


@pytest.mark.parametrize("tiny", [5e-324, 1e-300, 1e-12])
def test_near_zero_counts_as_null_island(tiny):
    """An exact comparison let denormals and float residue through."""
    assert _pts([(tiny, tiny)] * 5).select(plc.detect("x", "y")).item() == "unknown"


def test_report_surfaces_how_much_was_dropped():
    df = _pts([WGS_PT] * 5 + [(0.0, 0.0)] * 95)
    assert "null-island=95/100" in df.select(plc.detect_report("x", "y")).item()


# --- input validation --------------------------------------------------------


def test_boolean_columns_are_rejected():
    """True/False cast to 1.0/0.0, which sits inside the lon/lat box."""
    df = pl.DataFrame({"x": [True, False] * 5, "y": [True, True] * 5})
    with pytest.raises(Exception, match="numeric coordinates"):
        df.select(plc.detect("x", "y"))


@pytest.mark.parametrize(
    "value",
    [
        datetime.date(2024, 1, 1),
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        datetime.timedelta(days=1),
        datetime.time(12, 0),
    ],
)
def test_temporal_columns_are_rejected(value):
    """Temporal types cast to an epoch offset that lands in a real CRS range."""
    df = pl.DataFrame({"x": [value] * 5, "y": [value] * 5})
    with pytest.raises(Exception, match="numeric coordinates"):
        df.select(plc.detect("x", "y"))


@pytest.mark.parametrize(
    "dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
)
def test_numeric_dtypes_are_accepted(dtype):
    df = pl.DataFrame(
        {
            "x": pl.Series([121000], dtype=dtype),
            "y": pl.Series([487000], dtype=dtype),
        }
    )
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


def test_same_column_for_both_axes_is_rejected():
    """Always a caller mistake, and cheap to catch."""
    df = pl.DataFrame({"x": [121000.0, 92000.0]})
    with pytest.raises(Exception, match="same values"):
        df.select(plc.detect("x", "x"))


@pytest.mark.parametrize("value", [0.0, -999.0, 1e12])
def test_constant_columns_are_not_mistaken_for_a_duplicate_axis(value):
    """Sentinel data legitimately holds the same value on both axes.

    Rejecting it would refuse exactly the input this tool exists to diagnose.
    """
    df = pl.DataFrame({"x": [value] * 5, "y": [value] * 5})
    df.select(plc.detect("x", "y"))  # must not raise


# --- report detail -----------------------------------------------------------


def test_report_counts_rows_a_fraction_would_round_away():
    """One stray row in 100000 rounds to 0.00 but must stay countable."""
    df = pl.DataFrame(
        {
            "x": [121000.0] * 99_999 + [4.9],
            "y": [487000.0] * 99_999 + [52.4],
        }
    )
    got = df.select(plc.detect_report("x", "y")).item()
    assert "EPSG:4326=0.00/0.00/1" in got


def test_unverifiable_axis_order_is_flagged():
    """Dutch coordinates are valid WGS84 either way round."""
    df = pl.DataFrame({"x": [4.9041, 5.12], "y": [52.3676, 52.09]})
    assert "axis-order=unverifiable" in df.select(plc.detect_report("x", "y")).item()


def test_verifiable_axis_order_is_not_flagged():
    """A longitude beyond +/-90 settles the order, so there is nothing to warn about."""
    df = pl.DataFrame({"x": [4.9, -74.0, 151.2], "y": [52.4, 40.7, -33.9]})
    got = df.select(plc.detect_report("x", "y")).item()
    assert "axis-order" not in got and "swapped" not in got


# --- regressions from the 0.1.5 input guard ----------------------------------


def test_decimal_does_not_abort_the_process():
    """0.1.5 aborted with SIGABRT here, which no caller could catch.

    polars was built without dtype-decimal, so the cast panicked inside a
    non-unwinding boundary. The feature is now enabled.
    """
    from decimal import Decimal

    df = pl.DataFrame(
        {
            "x": pl.Series([Decimal("121000.0000")], dtype=pl.Decimal(18, 4)),
            "y": pl.Series([Decimal("487000.0000")], dtype=pl.Decimal(18, 4)),
        }
    )
    assert df.select(plc.detect("x", "y")).item() == "EPSG:28992"


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ],
)
def test_every_integer_width_works(dtype):
    """Small widths panicked in 0.1.5: polars lacked dtype-i8/i16/u8/u16."""
    df = pl.DataFrame(
        {
            "x": pl.Series([5], dtype=dtype),
            "y": pl.Series([52], dtype=dtype),
        }
    )
    assert df.select(plc.detect("x", "y")).item() == "EPSG:4326"


def test_group_by_agg_works():
    """0.1.5 rejected this: Polars hands a plugin unnamed series inside an
    aggregation, so a name-based duplicate check saw two empty names."""
    df = pl.DataFrame(
        {
            "src": ["nl", "nl", "uk", "uk"],
            "lon": [121000.0, 92000.0, 530000.0, 325000.0],
            "lat": [487000.0, 437000.0, 180000.0, 673000.0],
        }
    )
    got = df.group_by("src", maintain_order=True).agg(
        plc.detect("lon", "lat").alias("crs")
    )
    assert got["crs"].to_list() == ["EPSG:28992", "EPSG:27700"]


def test_over_works():
    df = pl.DataFrame(
        {
            "src": ["nl", "nl", "uk", "uk"],
            "lon": [121000.0, 92000.0, 530000.0, 325000.0],
            "lat": [487000.0, 437000.0, 180000.0, 673000.0],
        }
    )
    got = df.select(plc.detect("lon", "lat").over("src")).to_series().to_list()
    assert got == ["EPSG:28992", "EPSG:28992", "EPSG:27700", "EPSG:27700"]


def test_duplicate_axes_compared_on_data_not_name():
    """Names are unreliable in both directions, so compare the values."""
    df = pl.DataFrame({"x": [121000.0, 92000.0], "y": [487000.0, 437000.0]})

    # two different columns sharing an alias must be accepted
    assert (
        df.select(plc.detect(pl.col("x").alias("a"), pl.col("y").alias("a"))).item()
        == "EPSG:28992"
    )

    # two literals are both named "literal" but hold different values
    assert (
        df.select(plc.detect(pl.lit(155000.0), pl.lit(463000.0))).item() == "EPSG:28992"
    )

    # the same data under a different name must still be rejected
    with pytest.raises(Exception, match="same values"):
        df.select(plc.detect("x", pl.col("x").alias("y")))

    with pytest.raises(Exception, match="same values"):
        df.select(plc.detect("x", "x"))
