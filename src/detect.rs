//! Whole-column detection: which system is consistent with all the rows?
//!
//! Elimination rather than voting. Every system starts as a candidate and each
//! row rules out the ones whose range it falls outside, so more rows narrow the
//! answer instead of producing more disagreement.

use polars::prelude::*;

use crate::crs::{match_mask, CRS_DEFS, UNKNOWN};

/// A system must contain at least this fraction of the rows to stay a
/// candidate.
///
/// Real data has outliers: typos, sentinels, points just outside an official
/// area of use. Requiring every row to match would let one bad row discard the
/// correct answer.
pub const MATCH_THRESHOLD: f64 = 0.95;

/// Fraction of finite rows each CRS contains, and how many rows were counted.
///
/// Nulls and NaNs are skipped rather than counted as misses: absent data is not
/// evidence against a system.
pub fn match_fractions(x: &Float64Chunked, y: &Float64Chunked) -> (Vec<f64>, usize) {
    let mut hits = vec![0usize; CRS_DEFS.len()];
    let mut seen = 0usize;

    for (a, b) in x.into_iter().zip(y.into_iter()) {
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a.is_nan() || b.is_nan() {
            continue;
        }
        seen += 1;
        let mask = match_mask(a, b);
        for (i, h) in hits.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                *h += 1;
            }
        }
    }

    if seen == 0 {
        return (vec![0.0; CRS_DEFS.len()], 0);
    }
    (hits.iter().map(|h| *h as f64 / seen as f64).collect(), seen)
}

/// The narrowest system meeting the threshold.
pub fn best(fracs: &[f64], seen: usize) -> String {
    if seen == 0 {
        return UNKNOWN.to_string();
    }
    fracs
        .iter()
        .position(|f| *f >= MATCH_THRESHOLD)
        .map(|i| CRS_DEFS[i].code.to_string())
        .unwrap_or_else(|| UNKNOWN.to_string())
}

/// Every system meeting the threshold, pipe-joined.
pub fn surviving(fracs: &[f64], seen: usize) -> String {
    let survivors: Vec<&str> = fracs
        .iter()
        .enumerate()
        .filter(|(_, f)| **f >= MATCH_THRESHOLD)
        .map(|(i, _)| CRS_DEFS[i].code)
        .collect();

    if seen == 0 || survivors.is_empty() {
        UNKNOWN.to_string()
    } else {
        survivors.join("|")
    }
}

/// Match fraction per system, highest first, as `"EPSG:28992=1.00|..."`.
pub fn report(fracs: &[f64], seen: usize) -> String {
    if seen == 0 {
        return UNKNOWN.to_string();
    }
    let mut parts: Vec<(usize, f64)> = fracs.iter().copied().enumerate().collect();
    parts.retain(|(_, f)| *f > 0.0);
    parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if parts.is_empty() {
        UNKNOWN.to_string()
    } else {
        parts
            .iter()
            .map(|(i, f)| format!("{}={:.2}", CRS_DEFS[*i].code, f))
            .collect::<Vec<_>>()
            .join("|")
    }
}
