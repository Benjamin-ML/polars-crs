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

/// Both coordinate columns must be Float64. Integer grid references are common
/// in the wild, so say what to do rather than just failing.
fn coord_pair(inputs: &[Series]) -> PolarsResult<(&Float64Chunked, &Float64Chunked)> {
    for (i, s) in inputs.iter().take(2).enumerate() {
        if s.dtype() != &DataType::Float64 {
            polars_bail!(
                InvalidOperation:
                "polars-crs expects Float64 coordinates, got {} for argument {}. \
                 Cast first, e.g. pl.col('x').cast(pl.Float64).",
                s.dtype(), i
            );
        }
    }
    Ok((inputs[0].f64()?, inputs[1].f64()?))
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

    let out = par_map_str(x, y, "guess", |a, b| first_match(a, b).unwrap_or(UNKNOWN));
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn candidates(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
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
    let (fracs, seen) = match_fractions(x, y);
    Ok(scalar_string("detect", best(&fracs, seen)))
}

#[polars_expr(output_type=String)]
fn detect_candidates(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (fracs, seen) = match_fractions(x, y);
    Ok(scalar_string("detect_candidates", surviving(&fracs, seen)))
}

#[polars_expr(output_type=String)]
fn detect_report(inputs: &[Series]) -> PolarsResult<Series> {
    let (x, y) = coord_pair(inputs)?;
    let (fracs, seen) = match_fractions(x, y);
    Ok(scalar_string("detect_report", report(&fracs, seen)))
}
