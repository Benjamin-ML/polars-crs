//! Whole-column detection: which system is consistent with all the rows?
//!
//! Elimination rather than voting. Every system starts as a candidate and each
//! row rules out the ones whose range it falls outside, so more rows narrow the
//! answer instead of producing more disagreement.

use polars::prelude::*;
use rayon::prelude::*;

use crate::crs::{match_mask, CRS_DEFS, MIXED, SWAPPED, UNKNOWN, WGS84};
use crate::parallel::PARALLEL_THRESHOLD;

/// A system must contain at least this fraction of the rows to stay a
/// candidate.
///
/// Real data has outliers: typos, sentinels, points just outside an official
/// area of use. Requiring every row to match would let one bad row discard the
/// correct answer.
pub const MATCH_THRESHOLD: f64 = 0.95;

/// A single system must be the narrowest match for at least this share of the
/// rows to be reported on its own.
///
/// Not 1.0, because ranges overlap: some British points genuinely fall inside
/// the Dutch range, so even a pure column has a minority preferring a neighbour.
pub const DOMINANT: f64 = 0.80;

/// A system whose narrowest-match share reaches this is listed as a component
/// of a mixture rather than treated as noise.
pub const MIXTURE_MIN: f64 = 0.05;

/// Coordinates this close to (0, 0) are treated as Null Island sentinels.
/// An exact comparison lets denormals and the residue of float arithmetic slip
/// through, and nothing real is measured to this precision.
pub const ZERO_EPS: f64 = 1e-9;

/// Per-column statistics.
pub struct Stats {
    /// Fraction of rows each system's range contains. Systems overlap, so these
    /// do not sum to 1.
    pub contains: Vec<f64>,
    /// Fraction of rows for which each system is the NARROWEST match. These do
    /// sum to 1 (less any rows matching nothing), and are what distinguishes a
    /// mixture from a single system.
    pub narrowest: Vec<f64>,
    /// Fraction of rows that would fit WGS84 if x and y were exchanged. A high
    /// value with a low `contains[WGS84]` means the axes are reversed.
    pub swapped: f64,
    /// Rows counted, excluding nulls, NaNs and Null Island sentinels.
    pub seen: usize,
    /// Row count for which each system is the narrowest match. A fraction
    /// rounds a handful of stray rows to 0.00; the count does not.
    pub narrowest_n: Vec<usize>,
    /// Rows dropped as Null Island sentinels.
    pub sentinels: usize,
}

/// Raw tallies for a slice of rows. Counts, so slices merge by addition.
#[derive(Default)]
struct Tally {
    contains: Vec<usize>,
    narrowest: Vec<usize>,
    swapped: usize,
    seen: usize,
    sentinels: usize,
}

impl Tally {
    fn new() -> Self {
        Tally {
            contains: vec![0; CRS_DEFS.len()],
            narrowest: vec![0; CRS_DEFS.len()],
            ..Default::default()
        }
    }

    fn merge(mut self, other: Tally) -> Self {
        for (a, b) in self.contains.iter_mut().zip(other.contains) {
            *a += b;
        }
        for (a, b) in self.narrowest.iter_mut().zip(other.narrowest) {
            *a += b;
        }
        self.swapped += other.swapped;
        self.seen += other.seen;
        self.sentinels += other.sentinels;
        self
    }
}

fn tally_slice(x: &Float64Chunked, y: &Float64Chunked) -> Tally {
    let mut t = Tally::new();
    for (a, b) in x.into_iter().zip(y) {
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a.is_nan() || b.is_nan() {
            continue;
        }
        if a.abs() < ZERO_EPS && b.abs() < ZERO_EPS {
            t.sentinels += 1;
            continue;
        }
        t.seen += 1;
        let mask = match_mask(a, b);
        for (i, c) in t.contains.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                *c += 1;
            }
        }
        if mask != 0 {
            t.narrowest[mask.trailing_zeros() as usize] += 1;
        }
        if CRS_DEFS[WGS84].contains(b, a) {
            t.swapped += 1;
        }
    }
    t
}

/// Nulls and NaNs are skipped rather than counted as misses: absent data is not
/// evidence against a system.
///
/// Anything within [`ZERO_EPS`] of (0, 0) is skipped too. Null Island is open
/// ocean, so a zero pair is almost always a missing value encoded as a number.
/// Counting it would let a column of missing data read as confident WGS84.
///
/// Rows are tallied in parallel. The accumulators are counts, so slices merge
/// by addition and the result does not depend on how the work was divided.
pub fn match_fractions(x: &Float64Chunked, y: &Float64Chunked) -> Stats {
    let n = CRS_DEFS.len();
    let rows = x.len();

    let t = if rows < PARALLEL_THRESHOLD {
        tally_slice(x, y)
    } else {
        let threads = rayon::current_num_threads().max(1);
        let chunk = rows.div_ceil(threads);
        (0..threads)
            .into_par_iter()
            .map(|i| {
                let start = i * chunk;
                if start >= rows {
                    return Tally::new();
                }
                let len = chunk.min(rows - start);
                tally_slice(&x.slice(start as i64, len), &y.slice(start as i64, len))
            })
            .reduce(Tally::new, Tally::merge)
    };

    if t.seen == 0 {
        return Stats {
            contains: vec![0.0; n],
            narrowest: vec![0.0; n],
            swapped: 0.0,
            seen: 0,
            narrowest_n: vec![0; n],
            sentinels: t.sentinels,
        };
    }
    let seen = t.seen as f64;
    Stats {
        contains: t.contains.iter().map(|c| *c as f64 / seen).collect(),
        narrowest: t.narrowest.iter().map(|c| *c as f64 / seen).collect(),
        swapped: t.swapped as f64 / seen,
        seen: t.seen,
        narrowest_n: t.narrowest,
        sentinels: t.sentinels,
    }
}

/// True when the column is WGS84 with x and y exchanged.
///
/// Only detectable when some row carries a longitude beyond +/-90, which cannot
/// be a latitude. A dataset confined to low latitudes and longitudes, the
/// Netherlands for instance, fits WGS84 either way round and no test on the
/// values can separate the two.
fn axes_reversed(st: &Stats) -> bool {
    st.swapped >= MATCH_THRESHOLD && st.contains[WGS84] < MATCH_THRESHOLD
}

/// True when the column fits WGS84 both as given and with x and y exchanged.
///
/// Every value then sits within +/-90, where a coordinate is valid either way
/// round. Nothing in the numbers can settle the order, so say so rather than
/// returning EPSG:4326 as though it had been checked.
fn axis_order_unverifiable(st: &Stats) -> bool {
    st.contains[WGS84] >= DOMINANT && st.swapped >= DOMINANT
}

/// Index of the system that is the narrowest match for the largest share of
/// rows. Ties go to the narrower range, which is the earlier entry.
fn dominant(st: &Stats) -> Option<usize> {
    st.narrowest
        .iter()
        .enumerate()
        .filter(|(_, f)| **f > 0.0)
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(i, _)| i)
}

/// Systems that together account for the column when no single one dominates.
fn components(st: &Stats) -> Option<Vec<&'static str>> {
    let parts: Vec<usize> = (0..CRS_DEFS.len())
        .filter(|i| st.narrowest[*i] >= MIXTURE_MIN)
        .collect();
    let covered: f64 = parts.iter().map(|i| st.narrowest[*i]).sum();
    if parts.len() >= 2 && covered >= DOMINANT {
        Some(parts.iter().map(|i| CRS_DEFS[*i].code).collect())
    } else {
        None
    }
}

/// The verdict for a column.
///
/// Decided entirely on the narrowest-match shares, which sum to one. Those are
/// the only exclusive evidence available; how many rows a range merely
/// *contains* is diagnostic and must never decide the answer. Letting
/// containment decide is what made a broad range win a column it had no
/// coordinates in, first for mixed columns and then again inside the band where
/// no single system reached the threshold.
pub fn best(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    if axes_reversed(st) {
        return format!("{}:{}", SWAPPED, CRS_DEFS[WGS84].code);
    }
    if let Some(i) = dominant(st) {
        if st.narrowest[i] >= DOMINANT {
            return CRS_DEFS[i].code.to_string();
        }
    }
    match components(st) {
        Some(parts) => format!("{}:{}", MIXED, parts.join("|")),
        None => UNKNOWN.to_string(),
    }
}

/// Like [`best`], but a single system is reported alongside anything else that
/// is the narrowest match for a meaningful share.
pub fn surviving(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    if axes_reversed(st) {
        return format!("{}:{}", SWAPPED, CRS_DEFS[WGS84].code);
    }
    let parts: Vec<&str> = (0..CRS_DEFS.len())
        .filter(|i| st.narrowest[*i] >= MIXTURE_MIN)
        .map(|i| CRS_DEFS[i].code)
        .collect();
    if parts.is_empty() {
        UNKNOWN.to_string()
    } else {
        parts.join("|")
    }
}

/// Per-system detail: `narrowest/contains`.
///
/// `narrowest` is the share of rows for which the system is the most specific
/// match, and these sum to one. `contains` is the share merely inside its
/// range; ranges overlap, so these do not. A system reading `0.00/1.00`
/// contains every row and is preferred by none, which means its range is
/// swallowing the real answer.
pub fn report(st: &Stats) -> String {
    if st.seen == 0 {
        return if st.sentinels > 0 {
            format!("{}|null-island={}/{}", UNKNOWN, st.sentinels, st.sentinels)
        } else {
            UNKNOWN.to_string()
        };
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
        return UNKNOWN.to_string();
    }
    let mut out: Vec<String> = parts
        .iter()
        .map(|(i, c, nrw)| {
            format!(
                "{}={:.2}/{:.2}/{}",
                CRS_DEFS[*i].code, *nrw, *c, st.narrowest_n[*i]
            )
        })
        .collect();
    if axes_reversed(st) {
        out.push(format!("{}={:.2}", SWAPPED, st.swapped));
    } else if axis_order_unverifiable(st) {
        out.push("axis-order=unverifiable".to_string());
    }
    if st.sentinels > 0 {
        out.push(format!(
            "null-island={}/{}",
            st.sentinels,
            st.sentinels + st.seen
        ));
    }
    out.join("|")
}
