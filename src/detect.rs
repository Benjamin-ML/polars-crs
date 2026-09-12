//! Whole-column detection: which system is consistent with all the rows?
//!
//! Elimination rather than voting. Every system starts as a candidate and each
//! row rules out the ones whose range it falls outside, so more rows narrow the
//! answer instead of producing more disagreement.

use polars::prelude::*;

use crate::crs::{match_mask, CRS_DEFS, MIXED, UNKNOWN};

/// A system must contain at least this fraction of the rows to stay a
/// candidate.
///
/// Real data has outliers: typos, sentinels, points just outside an official
/// area of use. Requiring every row to match would let one bad row discard the
/// correct answer.
pub const MATCH_THRESHOLD: f64 = 0.95;

/// A system whose narrowest-match share reaches this counts as a component of
/// a mixture, rather than noise from overlapping ranges.
pub const MIXTURE_MIN: f64 = 0.10;

/// Per-column statistics.
pub struct Stats {
    /// Fraction of rows each system's range contains. Systems overlap, so these
    /// do not sum to 1.
    pub contains: Vec<f64>,
    /// Fraction of rows for which each system is the NARROWEST match. These do
    /// sum to 1 (less any rows matching nothing), and are what distinguishes a
    /// mixture from a single system.
    pub narrowest: Vec<f64>,
    /// Rows counted, excluding nulls and NaNs.
    pub seen: usize,
}

/// Nulls and NaNs are skipped rather than counted as misses: absent data is not
/// evidence against a system.
pub fn match_fractions(x: &Float64Chunked, y: &Float64Chunked) -> Stats {
    let n = CRS_DEFS.len();
    let mut contains = vec![0usize; n];
    let mut narrowest = vec![0usize; n];
    let mut seen = 0usize;

    for (a, b) in x.into_iter().zip(y.into_iter()) {
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a.is_nan() || b.is_nan() {
            continue;
        }
        seen += 1;
        let mask = match_mask(a, b);
        for (i, c) in contains.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                *c += 1;
            }
        }
        if mask != 0 {
            narrowest[mask.trailing_zeros() as usize] += 1;
        }
    }

    if seen == 0 {
        return Stats {
            contains: vec![0.0; n],
            narrowest: vec![0.0; n],
            seen: 0,
        };
    }
    Stats {
        contains: contains.iter().map(|c| *c as f64 / seen as f64).collect(),
        narrowest: narrowest.iter().map(|c| *c as f64 / seen as f64).collect(),
        seen,
    }
}

/// Systems that together account for the column when no single one does.
///
/// A broad range can contain two disjoint groups of coordinates and score 1.00
/// while each real system scores only its own share and fails the threshold.
/// Reporting the broad range as the answer names a system that is not present
/// in the data at all, so detect that case and report the components instead.
///
/// Returns None unless at least two systems narrower than `winner` each hold a
/// meaningful share of the narrowest matches AND together account for the
/// column.
fn mixture(st: &Stats, winner: usize) -> Option<Vec<&'static str>> {
    let covered: f64 = st.narrowest[..winner].iter().sum();
    if covered < MATCH_THRESHOLD {
        return None;
    }
    let parts: Vec<&'static str> = (0..winner)
        .filter(|i| st.narrowest[*i] >= MIXTURE_MIN)
        .map(|i| CRS_DEFS[i].code)
        .collect();
    (parts.len() >= 2).then_some(parts)
}

/// The narrowest system meeting the threshold, or a description of the mixture
/// when several systems together account for the column.
pub fn best(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    match st.contains.iter().position(|f| *f >= MATCH_THRESHOLD) {
        None => UNKNOWN.to_string(),
        Some(w) => match mixture(st, w) {
            Some(parts) => format!("{}:{}", MIXED, parts.join("|")),
            None => CRS_DEFS[w].code.to_string(),
        },
    }
}

/// Every system meeting the threshold, pipe-joined.
pub fn surviving(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    if let Some(w) = st.contains.iter().position(|f| *f >= MATCH_THRESHOLD) {
        if let Some(parts) = mixture(st, w) {
            return format!("{}:{}", MIXED, parts.join("|"));
        }
    }
    let survivors: Vec<&str> = st
        .contains
        .iter()
        .enumerate()
        .filter(|(_, f)| **f >= MATCH_THRESHOLD)
        .map(|(i, _)| CRS_DEFS[i].code)
        .collect();

    if survivors.is_empty() {
        UNKNOWN.to_string()
    } else {
        survivors.join("|")
    }
}

/// Per-system detail: `contains` is the fraction of rows inside each range,
/// `narrowest` the fraction for which it is the most specific match. Two
/// systems at roughly 0.50 narrowest is the signature of a mixed column.
pub fn report(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    let mut parts: Vec<(usize, f64, f64)> = (0..CRS_DEFS.len())
        .map(|i| (i, st.contains[i], st.narrowest[i]))
        .filter(|(_, c, _)| *c > 0.0)
        .collect();
    parts.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    if parts.is_empty() {
        UNKNOWN.to_string()
    } else {
        parts
            .iter()
            .map(|(i, c, nrw)| format!("{}={:.2}/{:.2}", CRS_DEFS[*i].code, *nrw, *c))
            .collect::<Vec<_>>()
            .join("|")
    }
}
