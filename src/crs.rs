//! Coordinate reference system definitions and per-point range matching.
//!
//! Nothing here allocates. These run once per row, where a heap allocation
//! would dominate the runtime.

use std::sync::OnceLock;

/// A coordinate reference system described by the numeric range its
/// coordinates occupy.
pub struct Crs {
    pub code: &'static str,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Crs {
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }
}

/// Ordered NARROWEST RANGE FIRST. A narrower range is more informative, so
/// when several match, the first is the most specific answer. This ordering is
/// load-bearing: `guess` and `detect` both take the first survivor.
///
/// Bounds are derived from PROJ: each system's official area of use, sampled
/// on a 200x200 grid and transformed into projected coordinates, then padded.
pub const CRS_DEFS: &[Crs] = &[
    Crs {
        code: "EPSG:4326",
        x_min: -180.0,
        x_max: 180.0,
        y_min: -90.0,
        y_max: 90.0,
    },
    Crs {
        code: "EPSG:28992",
        x_min: -1000.0,
        x_max: 290000.0,
        y_min: 300000.0,
        y_max: 640000.0,
    },
    Crs {
        code: "EPSG:27700",
        x_min: -110000.0,
        x_max: 690000.0,
        y_min: -20000.0,
        y_max: 1260000.0,
    },
    Crs {
        code: "EPSG:3857",
        x_min: -20037508.34,
        x_max: 20037508.34,
        y_min: -20048966.10,
        y_max: 20048966.10,
    },
];

pub const UNKNOWN: &str = "unknown";

/// Prefix for a column holding coordinates from more than one system.
pub const MIXED: &str = "mixed";

/// Prefix for a column whose x and y are the wrong way round.
pub const SWAPPED: &str = "swapped";

/// Index of the WGS84 entry in [`CRS_DEFS`].
pub const WGS84: usize = 0;

/// Index of the Dutch RD New entry in [`CRS_DEFS`].
pub const RD_NEW: usize = 1;

/// Bitmask of [`CRS_DEFS`] entries whose range contains the point.
///
/// NaN fails every comparison, so NaN input yields mask 0.
#[inline]
pub fn match_mask(x: f64, y: f64) -> u8 {
    let mut mask = 0u8;
    for (i, c) in CRS_DEFS.iter().enumerate() {
        if c.contains(x, y) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Narrowest matching system, or None. Borrows a static str -- no allocation.
#[inline]
pub fn first_match(x: f64, y: f64) -> Option<&'static str> {
    CRS_DEFS.iter().find(|c| c.contains(x, y)).map(|c| c.code)
}

/// Pipe-joined label for every possible mask, built once on first use.
///
/// With four systems there are only 16 distinct answers, so we render them
/// once instead of formatting a string per row.
pub fn mask_labels() -> &'static [String; 16] {
    static LABELS: OnceLock<[String; 16]> = OnceLock::new();
    LABELS.get_or_init(|| {
        std::array::from_fn(|mask| {
            let hits: Vec<&str> = CRS_DEFS
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, c)| c.code)
                .collect();
            if hits.is_empty() {
                UNKNOWN.to_string()
            } else {
                hits.join("|")
            }
        })
    })
}
