"""Generate realistic benchmark data: real coordinates in real systems."""

import sys

import numpy as np
import polars as pl
from pyproj import Transformer

AREAS = {
    "EPSG:28992": (3.2, 7.2, 50.75, 53.7),
    "EPSG:27700": (-7.6, 1.8, 49.9, 60.9),
    "EPSG:3857": (-180, 180, -85, 85),
    "EPSG:4326": (-180, 180, -90, 90),
}


def make(n: int, code: str = "EPSG:28992", seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    lon_lo, lon_hi, lat_lo, lat_hi = AREAS[code]
    lon = rng.uniform(lon_lo, lon_hi, n)
    lat = rng.uniform(lat_lo, lat_hi, n)
    if code == "EPSG:4326":
        x, y = lon, lat
    else:
        tf = Transformer.from_crs("EPSG:4326", code, always_xy=True)
        x, y = tf.transform(lon, lat)
    return pl.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    df = make(n)
    path = f"data/bench_{n}.parquet"
    df.write_parquet(path)
    print(f"{path}  {n:,} rows  {df.estimated_size('mb'):.1f} MB in memory")
