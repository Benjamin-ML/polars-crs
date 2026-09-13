#![allow(clippy::unused_unit)]
//! Polars expression entry points.
//!
//! These stay thin: validate dtypes, pull out the typed columns, delegate to
//! [`crate::crs`] or [`crate::detect`], wrap the result back into a Series.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::crs::{first_match, mask_labels, match_mask, CRS_DEFS, RD_NEW, UNKNOWN};
use crate::detect::{best, match_fractions, report, surviving};
use crate::parallel::par_map_str;

/// True when the two columns are almost certainly the same column passed twice.
///
/// Compared on the data, not on the name. Names are unreliable: Polars hands a
/// plugin unnamed series inside `group_by` and `over`, two different columns can
/// share an alias, and two literals are both called "literal".
///
/// A constant column is exempt. Sentinel data such as a column of (0, 0) or
/// (-999, -999) genuinely holds the same value on both axes, and rejecting it
/// would refuse input this tool exists to diagnose. Two real coordinate columns
/// never agree row-for-row across varying values.
///
/// Single pass, stopping at the first difference, so for genuinely different
/// columns it costs almost nothing.
fn same_column_twice(x: &Float64Chunked, y: &Float64Chunked) -> bool {
    if x.len() != y.len() || x.is_empty() {
        return false;
    }
    let first = x.get(0);
    let mut varies = false;
    for (a, b) in x.into_iter().zip(y) {
        if a != b {
            return false;
        }
        if a != first {
            varies = true;
        }
    }
    varies
}

/// Validate and coerce the two coordinate columns.
///
/// Anything castable to a float is not automatically a coordinate. Booleans
/// become 0.0 and 1.0, and temporal types become their epoch offset; both land
/// inside the lon/lat box and would produce a confident wrong answer from a
/// column nobody meant as coordinates.
fn coord_pair(inputs: &[Series]) -> PolarsResult<(Float64Chunked, Float64Chunked)> {
    if inputs.len() < 2 {
        polars_bail!(
            InvalidOperation:
            "polars-crs needs two coordinate columns, got {}",
            inputs.len()
        );
    }

    let mut out = Vec::with_capacity(2);
    for (i, s) in inputs.iter().take(2).enumerate() {
        let dt = s.dtype();
        // is_numeric() is false for Boolean and for every temporal type, which
        // is exactly the set that casts to a plausible coordinate.
        if !dt.is_numeric() {
            polars_bail!(
                InvalidOperation:
                "polars-crs expects numeric coordinates, got {} for argument {}. \
                 Booleans and temporal types cast to numbers that fall inside the \
                 lon/lat range, so they are rejected rather than silently detected. \
                 Cast explicitly if the values really are coordinates.",
                dt, i
            );
        }
        let cast = s.cast(&DataType::Float64)?;
        out.push(cast.f64()?.clone());
    }

    let y = out.pop().expect("two columns were pushed");
    let x = out.pop().expect("two columns were pushed");

    // Passing one column as both axes is always a caller mistake.
    if same_column_twice(&x, &y) {
        polars_bail!(
            InvalidOperation:
            "polars-crs was given the same values as both x and y. \
             Pass two different coordinate columns."
        );
    }
    Ok((x, y))
}

/// A single-row String Series, for the aggregating expressions.
fn scalar_string(name: &'static str, value: String) -> Series {
    StringChunked::from_iter_values(PlSmallStr::from_static(name), std::iter::once(value))
        .into_series()
}

// --- elementwise ------------------------------------------------------------

#[polars_expr(output_type=Boolean)]
fn is_rd_new(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);
    let rd = &CRS_DEFS[RD_NEW];

    let out: BooleanChunked = x
        .into_iter()
        .zip(y.into_iter())
        .map(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => Some(rd.contains(a, b)),
            _ => None,
        })
        .collect_ca(PlSmallStr::from_static("is_rd_new"));

    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn guess(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);

    let out = par_map_str(x, y, "guess", |a, b| first_match(a, b).unwrap_or(UNKNOWN));
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn candidates(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);
    let labels = mask_labels();

    let out = par_map_str(x, y, "candidates", |a, b| {
        labels[match_mask(a, b) as usize].as_str()
    });
    Ok(out.into_series())
}

// --- aggregating ------------------------------------------------------------

#[polars_expr(output_type=String)]
fn detect(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);
    let st = match_fractions(x, y);
    Ok(scalar_string("detect", best(&st)))
}

#[polars_expr(output_type=String)]
fn detect_candidates(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);
    let st = match_fractions(x, y);
    Ok(scalar_string("detect_candidates", surviving(&st)))
}

#[polars_expr(output_type=String)]
fn detect_report(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (x, y) = (&x, &y);
    let st = match_fractions(x, y);
    Ok(scalar_string("detect_report", report(&st)))
}
