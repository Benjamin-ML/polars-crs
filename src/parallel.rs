//! Splitting elementwise work across threads.
//!
//! A plugin receives the whole column in one call and, unless it says
//! otherwise, processes it on a single thread. Native Polars expressions run
//! across every core, which is where their advantage over a naive plugin comes
//! from. Slicing the input and mapping the pieces in parallel closes that gap.

use polars::prelude::*;
use rayon::prelude::*;

/// Below this many rows, thread setup costs more than it saves.
pub const PARALLEL_THRESHOLD: usize = 100_000;

/// Map two Float64 columns to a String column, in parallel over slices.
///
/// `f` is called once per row and must be cheap and side-effect free. The
/// result is multi-chunk, which Polars handles natively.
pub fn par_map_str<F>(
    x: &Float64Chunked,
    y: &Float64Chunked,
    name: &'static str,
    f: F,
) -> StringChunked
where
    F: Fn(f64, f64) -> &'static str + Sync + Send,
{
    let n = x.len();

    let map_slice = |x: &Float64Chunked, y: &Float64Chunked| -> StringChunked {
        x.into_iter()
            .zip(y)
            .map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(f(a, b)),
                _ => None,
            })
            .collect_ca(PlSmallStr::EMPTY)
    };

    if n < PARALLEL_THRESHOLD {
        let mut out = map_slice(x, y);
        out.rename(PlSmallStr::from_static(name));
        return out;
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(n_threads);

    let parts: Vec<StringChunked> = (0..n_threads)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk;
            if start >= n {
                return StringChunked::default();
            }
            let len = chunk.min(n - start);
            map_slice(&x.slice(start as i64, len), &y.slice(start as i64, len))
        })
        .collect();

    let mut out = StringChunked::default();
    for p in parts.iter().filter(|p| !p.is_empty()) {
        out.append(p).expect("appending String chunks cannot fail");
    }
    out.rename(PlSmallStr::from_static(name));
    out
}
