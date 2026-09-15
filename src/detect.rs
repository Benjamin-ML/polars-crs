//! Whole-column detection: which system is consistent with the rows?
//!
//! Decided on exclusive evidence only. For each row, the narrowest system that
//! contains it is that row's interpretation, and those shares sum to one. How
//! many rows a range merely *contains* is diagnostic and never decides
//! anything: a wide range contains whatever fits nothing else, so letting
//! containment decide makes it win columns it has no coordinates in.

use std::sync::OnceLock;

use polars::prelude::*;
use rayon::prelude::*;

use crate::crs::{match_mask, CRS_DEFS, MIXED, SWAPPED, UNKNOWN, WGS84};
use crate::parallel::PARALLEL_THRESHOLD;

/// A single interpretation must account for at least this share of the rows to
/// be reported on its own.
///
/// Not 1.0, because ranges overlap: some British points genuinely fall inside
/// the Dutch range, so even a pure column has a minority preferring a neighbour.
pub const DOMINANT: f64 = 0.80;

/// An interpretation holding at least this share is listed as a component of a
/// mixture rather than treated as noise.
pub const MIXTURE_MIN: f64 = 0.05;

/// A repeated point holding at least this share of the rows is a missing-value
/// placeholder, not a location.
pub const SENTINEL_MIN: f64 = 0.05;

/// Coordinates this close to (0, 0) are treated as Null Island sentinels.
/// An exact comparison lets denormals and the residue of float arithmetic slip
/// through, and nothing real is measured to this precision.
pub const ZERO_EPS: f64 = 1e-9;

/// How many candidate points to track while looking for placeholders.
///
/// Slots are held by frequency, not by arrival order. Filling them first-come
/// meant a file of varying coordinates exhausted them before a sentinel
/// appearing later in the column was ever seen.
const SENTINEL_SLOTS: usize = 64;

/// A repeated point holding at least this share is named in the report, even
/// when it is too small to change the verdict.
pub const SENTINEL_REPORT_MIN: f64 = 0.005;

/// At most this many rows per slice are examined when looking for repeated
/// points.
///
/// A placeholder is repeated by definition, so it turns up in a sample in
/// proportion to its share. Examining every row instead made the slot
/// bookkeeping dominate the runtime on varying data, where it can never find
/// anything: `detect` on 5,000,000 rows went from 15 ms to 59 ms.
const SENTINEL_SAMPLE: usize = 4096;

/// How a row is best read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// Narrowest system containing the row as given.
    AsGiven(usize),
    /// The row is a clean WGS84 coordinate with x and y exchanged.
    Swapped(usize),
    /// Reads identically either way round, so the order cannot be checked.
    Ambiguous,
    /// Inside no range at all.
    None,
}

/// Everything derived from one row, computed from two mask lookups.
///
/// Previously the reading and the transposition check each recomputed the
/// masks, so a row cost four range sweeps instead of two.
pub struct RowRead {
    pub reading: Reading,
    /// Containment as given, one bit per system.
    pub mask: u8,
    /// A strictly narrower system the row would land in if transposed.
    /// Suggestive only: a real coordinate in an overlapping region looks the
    /// same.
    pub transposed_to: Option<usize>,
}

/// Classify one row.
///
/// Only a WGS84 swap is decided here. Transposing lon/lat is self-diagnosing
/// once a longitude exceeds +/-90, which cannot be a latitude. Other
/// transpositions are not decidable from the values: Dutch RD turned round
/// lands inside British National Grid, but so does genuine southern England.
#[inline]
pub fn read_row(x: f64, y: f64) -> RowRead {
    let mg = match_mask(x, y);
    let ms = match_mask(y, x);
    let given = (mg != 0).then(|| mg.trailing_zeros() as usize);
    let swapped = (ms != 0).then(|| ms.trailing_zeros() as usize);
    let swaps_to_wgs84 = ms & (1 << WGS84) != 0;

    let reading = match given {
        Some(g) if g == WGS84 && swaps_to_wgs84 => Reading::Ambiguous,
        Some(g) if g == WGS84 => Reading::AsGiven(WGS84),
        _ if swaps_to_wgs84 => Reading::Swapped(WGS84),
        Some(g) => Reading::AsGiven(g),
        None => Reading::None,
    };
    let transposed_to = match (given, swapped) {
        (Some(g), Some(sw)) if sw < g => Some(sw),
        (None, Some(sw)) => Some(sw),
        _ => None,
    };
    RowRead {
        reading,
        mask: mg,
        transposed_to,
    }
}

/// The nine labels a single row can carry, rendered once.
///
/// Building these per row allocates, and a plugin that allocates per row loses
/// to a native expression chain outright. Indexed `[system][swapped]`.
fn row_labels() -> &'static [[String; 2]; 4] {
    static LABELS: OnceLock<[[String; 2]; 4]> = OnceLock::new();
    LABELS.get_or_init(|| {
        std::array::from_fn(|i| {
            [
                CRS_DEFS[i].code.to_string(),
                format!("{}:{}", SWAPPED, CRS_DEFS[i].code),
            ]
        })
    })
}

/// Label one row the same way [`best`] would label a column of it.
///
/// Used by the elementwise API so a row cannot be classified one way by
/// `guess` and another by `detect`. Column-level evidence is unavailable here:
/// a repeated placeholder is only recognisable across rows.
pub fn label_row(x: f64, y: f64) -> &'static str {
    if x.abs() < ZERO_EPS && y.abs() < ZERO_EPS {
        return UNKNOWN;
    }
    let l = row_labels();
    match read_row(x, y).reading {
        Reading::AsGiven(i) => l[i][0].as_str(),
        Reading::Ambiguous => l[WGS84][0].as_str(),
        Reading::Swapped(i) => l[i][1].as_str(),
        Reading::None => UNKNOWN,
    }
}

/// Raw tallies for a slice of rows. Counts, so slices merge by addition.
struct Tally {
    contains: Vec<usize>,
    narrowest: Vec<usize>,
    swapped: Vec<usize>,
    /// Rows that fit WGS84 identically either way round, so their axis order
    /// cannot be checked.
    order_ambiguous: usize,
    seen: usize,
    zeros: usize,
    /// Rows that would land in a narrower system if transposed. Suggestive
    /// only; genuine data in an overlapping region looks the same.
    transposable: Vec<usize>,
    /// Candidate placeholders: any repeated point. Whether one is missing data
    /// rather than a real location is decided afterwards, from what the rest of
    /// the column says.
    repeats: Vec<((u64, u64), usize)>,
}

impl Tally {
    fn new() -> Self {
        Tally {
            contains: vec![0; CRS_DEFS.len()],
            narrowest: vec![0; CRS_DEFS.len()],
            swapped: vec![0; CRS_DEFS.len()],
            order_ambiguous: 0,
            seen: 0,
            zeros: 0,
            transposable: vec![0; CRS_DEFS.len()],
            repeats: Vec::new(),
        }
    }

    /// Count repeated points, keeping exact counts for whatever is retained.
    ///
    /// When the slots are full, a slot still holding a single sighting is
    /// replaced; if every slot is holding a repeat, the new point is dropped.
    /// Counts are never inherited. An earlier version used Space-Saving, whose
    /// inheritance inflates one-off points until they look like repeats, which
    /// produced dozens of false candidates and an expensive recount.
    fn bump_repeat(&mut self, x: f64, y: f64) {
        let key = (x.to_bits(), y.to_bits());
        if let Some(slot) = self.repeats.iter_mut().find(|(k, _)| *k == key) {
            slot.1 += 1;
            return;
        }
        if self.repeats.len() < SENTINEL_SLOTS {
            self.repeats.push((key, 1));
            return;
        }
        if let Some(slot) = self.repeats.iter_mut().find(|(_, c)| *c == 1) {
            slot.0 = key;
            slot.1 = 1;
        }
    }

    fn merge(mut self, other: Tally) -> Self {
        for (a, b) in self.contains.iter_mut().zip(other.contains) {
            *a += b;
        }
        for (a, b) in self.narrowest.iter_mut().zip(other.narrowest) {
            *a += b;
        }
        for (a, b) in self.swapped.iter_mut().zip(other.swapped) {
            *a += b;
        }
        for (a, b) in self.transposable.iter_mut().zip(other.transposable) {
            *a += b;
        }
        self.order_ambiguous += other.order_ambiguous;
        self.seen += other.seen;
        self.zeros += other.zeros;
        // Merge with the same eviction rule as bump_repeat. Appending only
        // while free slots remain meant the first slice filled them and every
        // later slice's findings were discarded, so a sentinel in the tail of a
        // column was never seen.
        for (k, c) in other.repeats {
            if let Some(slot) = self.repeats.iter_mut().find(|(kk, _)| *kk == k) {
                slot.1 += c;
            } else if self.repeats.len() < SENTINEL_SLOTS {
                self.repeats.push((k, c));
            } else if let Some(slot) = self
                .repeats
                .iter_mut()
                .min_by_key(|(_, cc)| *cc)
                .filter(|(_, cc)| *cc < c)
            {
                *slot = (k, c);
            }
        }
        self
    }

    /// Remove the contribution of one repeated point.
    ///
    /// Every row at that point contributed identically, so the contribution is
    /// exactly `count` wherever its own reading landed.
    fn discount(&mut self, x: f64, y: f64, count: usize) {
        let r = read_row(x, y);
        for (i, c) in self.contains.iter_mut().enumerate() {
            if r.mask & (1 << i) != 0 {
                *c = c.saturating_sub(count);
            }
        }
        match r.reading {
            Reading::AsGiven(i) => self.narrowest[i] = self.narrowest[i].saturating_sub(count),
            Reading::Swapped(i) => self.swapped[i] = self.swapped[i].saturating_sub(count),
            Reading::Ambiguous => self.order_ambiguous = self.order_ambiguous.saturating_sub(count),
            Reading::None => {},
        }
        if let Some(i) = r.transposed_to {
            self.transposable[i] = self.transposable[i].saturating_sub(count);
        }
        self.seen = self.seen.saturating_sub(count);
        self.zeros += count;
    }
}

/// Exact count of each candidate point. Space-Saving only bounds them.
fn count_exactly(x: &Float64Chunked, y: &Float64Chunked, points: &[(f64, f64)]) -> Vec<usize> {
    let mut out = vec![0usize; points.len()];
    for (a, b) in x.into_iter().zip(y) {
        let (Some(a), Some(b)) = (a, b) else { continue };
        for (i, (px, py)) in points.iter().enumerate() {
            if a == *px && b == *py {
                out[i] += 1;
                break;
            }
        }
    }
    out
}

fn tally_slice(x: &Float64Chunked, y: &Float64Chunked) -> Tally {
    let mut t = Tally::new();
    let stride = (x.len() / SENTINEL_SAMPLE).max(1);
    let mut row = 0usize;
    for (a, b) in x.into_iter().zip(y) {
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a.is_nan() || b.is_nan() {
            continue;
        }
        if a.abs() < ZERO_EPS && b.abs() < ZERO_EPS {
            t.zeros += 1;
            continue;
        }
        t.seen += 1;
        if row.is_multiple_of(stride) {
            t.bump_repeat(a, b);
        }
        row += 1;

        let r = read_row(a, b);
        for (i, c) in t.contains.iter_mut().enumerate() {
            if r.mask & (1 << i) != 0 {
                *c += 1;
            }
        }
        // Only rows that landed in a catch-all range are candidates. A row
        // already read as a reversed WGS84 coordinate is explained; repeating
        // it as a transposition candidate says the same thing twice.
        if let Some(i) = r.transposed_to {
            if matches!(r.reading, Reading::AsGiven(j) if CRS_DEFS[j].catch_all) {
                t.transposable[i] += 1;
            }
        }
        match r.reading {
            Reading::AsGiven(i) => t.narrowest[i] += 1,
            Reading::Swapped(i) => t.swapped[i] += 1,
            // Reads the same either way round, so not evidence for the given
            // order. Held aside and apportioned afterwards by the rows that can
            // be told apart.
            Reading::Ambiguous => t.order_ambiguous += 1,
            Reading::None => {},
        }
    }
    t
}

/// Per-column statistics.
pub struct Stats {
    /// Fraction of rows each system's range contains, as given. Systems
    /// overlap, so these do not sum to 1. Diagnostic only.
    pub contains: Vec<f64>,
    /// Fraction of rows best read as each system, as given.
    pub narrowest: Vec<f64>,
    /// Fraction of rows best read as each system with the axes exchanged.
    pub swapped: Vec<f64>,
    /// Row counts behind `narrowest`. A fraction rounds a handful of stray rows
    /// to 0.00; the count does not.
    pub narrowest_n: Vec<usize>,
    /// Row counts behind `swapped`.
    pub swapped_n: Vec<usize>,
    /// Share of rows that read as WGS84 identically either way round.
    pub order_ambiguous: f64,
    /// Share of rows that would land in each narrower system if transposed.
    pub transposable: Vec<f64>,
    /// Rows counted, excluding nulls, NaNs and placeholders.
    pub seen: usize,
    /// Rows dropped as missing-value placeholders.
    pub sentinels: usize,
    /// The placeholder values that were dropped.
    pub sentinel_values: Vec<(f64, f64)>,
    /// Every repeated point seen, whether or not it changed the verdict.
    pub repeated_points: Vec<((f64, f64), usize)>,
}

/// Nulls and NaNs are skipped rather than counted as misses: absent data is not
/// evidence against a system.
///
/// Repeated points on the diagonal are dropped as placeholders once they hold
/// [`SENTINEL_MIN`] of the rows. `(0, 0)`, `(-999, -999)` and `(-9999, -9999)`
/// are all real coordinates somewhere, so nothing but their repetition marks
/// them as missing data.
///
/// Rows are tallied in parallel. The accumulators are counts, so slices merge
/// by addition and the result does not depend on how the work was divided.
pub fn match_fractions(x: &Float64Chunked, y: &Float64Chunked) -> Stats {
    let n = CRS_DEFS.len();
    let rows = x.len();

    let mut t = if rows < PARALLEL_THRESHOLD {
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

    // Decide which repeated points are missing-value placeholders.
    //
    // Space-Saving counts are upper bounds, so any candidate is recounted
    // exactly before it is named or acted on. Candidates are few, usually none,
    // so the recount is cheap and skipped entirely when there are no repeats
    // worth checking.
    let mut sentinel_values = Vec::new();
    let mut observed: Vec<((f64, f64), usize)> = Vec::new();
    if t.seen > 0 {
        // Counts here came from a sample, so compare against a scaled floor.
        // Candidates are recounted exactly below, which is what is acted on.
        let sampled = t.repeats.iter().map(|(_, c)| *c).sum::<usize>().max(1);
        let sample_floor = ((SENTINEL_REPORT_MIN * sampled as f64).ceil() as usize).max(2);
        let report_floor = ((SENTINEL_REPORT_MIN * t.seen as f64).ceil() as usize).max(2);
        let candidates: Vec<(f64, f64)> = t
            .repeats
            .iter()
            .filter(|(_, c)| *c >= sample_floor)
            .map(|((kx, ky), _)| (f64::from_bits(*kx), f64::from_bits(*ky)))
            .collect();

        if !candidates.is_empty() {
            let exact = count_exactly(x, y, &candidates);
            let drop_floor = ((SENTINEL_MIN * t.seen as f64).ceil() as usize).max(2);

            // what the column reads as, ignoring the candidates
            let rest_best = t
                .narrowest
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0)
                .max_by_key(|(_, n)| **n)
                .map(|(i, _)| i);

            for ((px, py), count) in candidates.iter().zip(exact) {
                if count < report_floor {
                    continue;
                }
                observed.push(((*px, *py), count));

                // A placeholder is a point the column disagrees with: either
                // nothing but a catch-all range contains it, or it reads as a
                // different system from everything around it. A real repeated
                // location agrees, so a depot in most of the rows is kept.
                let own = match read_row(*px, *py).reading {
                    Reading::None => None,
                    Reading::AsGiven(i) | Reading::Swapped(i) => Some(i),
                    Reading::Ambiguous => Some(WGS84),
                };
                let is_placeholder = match (own, rest_best) {
                    (None, _) => true,
                    (Some(o), Some(r)) => CRS_DEFS[o].catch_all || o != r,
                    (Some(o), None) => CRS_DEFS[o].catch_all,
                };
                if is_placeholder && count >= drop_floor {
                    t.discount(*px, *py, count);
                    sentinel_values.push((*px, *py));
                }
            }
        }
    }

    if t.seen == 0 {
        return Stats {
            contains: vec![0.0; n],
            narrowest: vec![0.0; n],
            swapped: vec![0.0; n],
            narrowest_n: vec![0; n],
            swapped_n: vec![0; n],
            order_ambiguous: 0.0,
            transposable: vec![0.0; n],
            seen: 0,
            sentinels: t.zeros,
            sentinel_values,
            repeated_points: observed,
        };
    }
    // Rows that read the same either way round go to whichever orientation the
    // decidable rows support, in proportion. With no decidable rows at all the
    // given order is assumed and the report says the order was not checked.
    let given = t.narrowest[WGS84];
    let swapped = t.swapped[WGS84];
    let decidable = given + swapped;
    if t.order_ambiguous > 0 {
        if decidable == 0 {
            t.narrowest[WGS84] += t.order_ambiguous;
        } else {
            let to_swapped =
                (t.order_ambiguous as f64 * swapped as f64 / decidable as f64).round() as usize;
            t.swapped[WGS84] += to_swapped;
            t.narrowest[WGS84] += t.order_ambiguous - to_swapped;
        }
    }

    let seen = t.seen as f64;
    Stats {
        contains: t.contains.iter().map(|c| *c as f64 / seen).collect(),
        narrowest: t.narrowest.iter().map(|c| *c as f64 / seen).collect(),
        swapped: t.swapped.iter().map(|c| *c as f64 / seen).collect(),
        order_ambiguous: t.order_ambiguous as f64 / seen,
        transposable: t.transposable.iter().map(|c| *c as f64 / seen).collect(),
        narrowest_n: t.narrowest,
        swapped_n: t.swapped,
        seen: t.seen,
        sentinels: t.zeros,
        sentinel_values,
        repeated_points: observed,
    }
}

/// Every interpretation with a non-zero share, as (label, share, count).
fn readings(st: &Stats) -> Vec<(String, f64, usize)> {
    let mut out: Vec<(String, f64, usize)> = Vec::new();
    for (i, crs) in CRS_DEFS.iter().enumerate() {
        if st.narrowest[i] > 0.0 {
            out.push((crs.code.to_string(), st.narrowest[i], st.narrowest_n[i]));
        }
        if st.swapped[i] > 0.0 {
            out.push((
                format!("{}:{}", SWAPPED, crs.code),
                st.swapped[i],
                st.swapped_n[i],
            ));
        }
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The verdict for a column.
pub fn best(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    let r = readings(st);
    if let Some((label, share, _)) = r.first() {
        if *share >= DOMINANT {
            return label.clone();
        }
    }
    let parts: Vec<&str> = r
        .iter()
        .filter(|(_, s, _)| *s >= MIXTURE_MIN)
        .map(|(l, _, _)| l.as_str())
        .collect();
    let covered: f64 = r
        .iter()
        .filter(|(_, s, _)| *s >= MIXTURE_MIN)
        .map(|(_, s, _)| s)
        .sum();
    if parts.len() >= 2 && covered >= DOMINANT {
        format!("{}:{}", MIXED, parts.join("|"))
    } else {
        UNKNOWN.to_string()
    }
}

/// Every interpretation holding a meaningful share.
pub fn surviving(st: &Stats) -> String {
    if st.seen == 0 {
        return UNKNOWN.to_string();
    }
    let parts: Vec<String> = readings(st)
        .into_iter()
        .filter(|(_, s, _)| *s >= MIXTURE_MIN)
        .map(|(l, _, _)| l)
        .collect();
    if parts.is_empty() {
        UNKNOWN.to_string()
    } else {
        parts.join("|")
    }
}

/// Per-interpretation detail: `LABEL=share/contains/rows`.
///
/// `share` is the fraction of rows best read this way, and these sum to one.
/// `contains` is the fraction merely inside the range as given; ranges overlap,
/// so these do not sum. A line reading `0.00/1.00` contains every row and is
/// preferred by none, which means its range is swallowing the real answer.
pub fn report(st: &Stats) -> String {
    let mut out: Vec<String> = Vec::new();

    if st.seen > 0 {
        for (label, share, count) in readings(st) {
            let idx = CRS_DEFS
                .iter()
                .position(|c| label.ends_with(c.code))
                .unwrap_or(0);
            let contains = if label.starts_with(SWAPPED) {
                share
            } else {
                st.contains[idx]
            };
            out.push(format!("{}={:.2}/{:.2}/{}", label, share, contains, count));
        }
        // systems that contain rows but are never the best reading
        for (i, (crs, c)) in CRS_DEFS.iter().zip(st.contains.iter()).enumerate() {
            if *c > 0.0 && st.narrowest[i] == 0.0 && st.swapped[i] == 0.0 {
                out.push(format!("{}=0.00/{:.2}/0", crs.code, c));
            }
        }
        if st.order_ambiguous >= DOMINANT {
            out.push("axis-order=unverifiable".to_string());
        }
        // A transposition candidate is only worth raising when it accounts for
        // the rows a catch-all range is currently holding. On a clean British
        // file about a tenth of rows would also fit the Dutch box transposed,
        // which is overlap noise on a correct answer rather than a finding.
        let container = st
            .narrowest
            .iter()
            .enumerate()
            .filter(|(i, n)| CRS_DEFS[*i].catch_all && **n > 0.0)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, n)| *n)
            .unwrap_or(0.0);
        if container >= MIXTURE_MIN {
            for (crs, share) in CRS_DEFS.iter().zip(st.transposable.iter()) {
                if *share >= 0.9 * container {
                    out.push(format!("transposed-candidate:{}={:.2}", crs.code, share));
                }
            }
        }
    }

    for ((px, py), count) in &st.repeated_points {
        let dropped = st.sentinel_values.contains(&(*px, *py));
        if !dropped {
            let label = if px == py {
                format!("{}", px)
            } else {
                format!("({},{})", px, py)
            };
            out.push(format!("repeated[{}]={}", label, count));
        }
    }

    if st.sentinels > 0 {
        let vals: Vec<String> = st
            .sentinel_values
            .iter()
            .map(|(a, b)| {
                if a == b {
                    format!("{}", a)
                } else {
                    format!("({},{})", a, b)
                }
            })
            .collect();
        let label = if vals.is_empty() {
            "0".to_string()
        } else {
            vals.join(",")
        };
        out.push(format!(
            "placeholder[{}]={}/{}",
            label,
            st.sentinels,
            st.sentinels + st.seen
        ));
    }

    if out.is_empty() {
        UNKNOWN.to_string()
    } else {
        out.join("|")
    }
}
