"""polars-crs - detect which coordinate reference system unlabelled x/y columns use.

Two ways to call everything:

    import polars_crs as plc
    df.with_columns(plc.detect("x", "y"))          # plain functions
    df.with_columns(pl.col("x").crs.detect("y"))   # namespace, reads like polars

See BUILD_PLAN.md for the detection method and its documented limitations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_crs._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_crs.typing import IntoExprColumn

LIB = Path(__file__).parent

__all__ = [
    "CRS_RANGES",
    "MATCH_THRESHOLD",
    "candidates",
    "detect",
    "detect_candidates",
    "detect_report",
    "guess",
    "is_rd_new",
]

#: A system must contain at least this fraction of rows to stay a candidate.
MATCH_THRESHOLD = 0.95

#: The ranges the detector uses, in priority order. Mirrors CRS_DEFS in
#: src/expressions.rs - keep the two in sync.
CRS_RANGES = {
    "EPSG:4326": {"name": "WGS84 lon/lat", "x": (-180.0, 180.0), "y": (-90.0, 90.0)},
    "EPSG:28992": {
        "name": "Dutch RD New",
        "x": (-1000.0, 290000.0),
        "y": (300000.0, 640000.0),
    },
    "EPSG:27700": {
        "name": "British National Grid",
        "x": (-110000.0, 690000.0),
        "y": (-20000.0, 1260000.0),
    },
    "EPSG:3857": {
        "name": "Web Mercator",
        "x": (-20037508.34, 20037508.34),
        "y": (-20048966.10, 20048966.10),
    },
}


def _col(e: IntoExprColumn) -> pl.Expr:
    """Normalise a column reference to an expression, without casting.

    Casting here would hide the input dtype from the Rust side, which needs to
    reject Booleans and temporal types: those cast to numbers inside the lon/lat
    range and would otherwise detect as a confident EPSG:4326.
    """
    if isinstance(e, str):
        return pl.col(e)
    if isinstance(e, pl.Expr):
        return e
    return pl.lit(e)


def is_rd_new(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """True where the point falls inside the Dutch RD New (EPSG:28992) range."""
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="is_rd_new",
        is_elementwise=True,
    )


def candidates(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """Every CRS whose range contains the point, pipe-joined.

    e.g. ``"EPSG:28992|EPSG:27700"``. ``"unknown"`` when nothing matches.
    Ranges genuinely overlap, so more than one answer is normal and honest.
    """
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="candidates",
        is_elementwise=True,
    )


def guess(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """Single best guess per row, by priority order. ``"unknown"`` if none match.

    Prefer :func:`candidates` when you need to know an answer was ambiguous.
    """
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="guess",
        is_elementwise=True,
    )


def detect(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """One verdict for the whole column.

    Starts with every CRS as a candidate and lets each row eliminate any whose
    range it falls outside. More rows narrow the answer, rather than producing
    more disagreement. Nulls and NaNs are skipped.

    Returns a single value, so this is an aggregation, not elementwise.
    """
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="detect",
        is_elementwise=False,
        returns_scalar=True,
    )


def detect_candidates(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """Like :func:`detect`, but reports every surviving candidate.

    Use when you need to see that the answer was ambiguous rather than take
    the narrowest one on trust.
    """
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="detect_candidates",
        is_elementwise=False,
        returns_scalar=True,
    )


def detect_report(x: IntoExprColumn, y: IntoExprColumn) -> pl.Expr:
    """Fraction of rows matching each system, e.g. ``"EPSG:28992=0.98|EPSG:3857=1.00"``.

    Shows how confident the verdict is rather than hiding it.
    """
    return register_plugin_function(
        args=[_col(x), _col(y)],
        plugin_path=LIB,
        function_name="detect_report",
        is_elementwise=False,
        returns_scalar=True,
    )


@pl.api.register_expr_namespace("crs")
class CrsNamespace:
    """Namespace so it reads like native Polars: ``pl.col("x").crs.detect("y")``."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def is_rd_new(self, y: IntoExprColumn) -> pl.Expr:
        return is_rd_new(self._expr, y)

    def candidates(self, y: IntoExprColumn) -> pl.Expr:
        return candidates(self._expr, y)

    def guess(self, y: IntoExprColumn) -> pl.Expr:
        return guess(self._expr, y)

    def detect(self, y: IntoExprColumn) -> pl.Expr:
        return detect(self._expr, y)

    def detect_candidates(self, y: IntoExprColumn) -> pl.Expr:
        return detect_candidates(self._expr, y)

    def detect_report(self, y: IntoExprColumn) -> pl.Expr:
        return detect_report(self._expr, y)
