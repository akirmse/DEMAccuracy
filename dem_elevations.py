#!/usr/bin/env python
"""Sample GLO-30, FABDEM, ALOS AW3D30, and NASADEM elevations at peak coordinates.

Reads a CSV with Lat/Long columns (WGS84), downloads the 1x1 degree DEM tile
covering each point (cached locally, reused if present), and writes a copy of
the CSV with the GLO-30 / FABDEM / ALOS / NASADEM columns filled in.

By default the reported heights are ellipsoidal (WGS84): the geoid undulation
at each point is added to the DEM's orthometric value, using the geoid model
each DEM is referenced to (EGM2008 for GLO-30/FABDEM, EGM96 for ALOS/NASADEM).
The undulation grids are the official GeoTIFFs from cdn.proj.org, cached like
the DEM tiles. Use --no-geoid for raw orthometric DEM values.

Tile sources (no authentication required):
  GLO-30  : Copernicus DEM GLO-30 COGs on AWS S3
  FABDEM  : University of Bristol V1-2. Distributed as 10x10 degree zips of
            1+ GB; only the needed 1-degree tile is extracted, via HTTP range
            requests (remotezip), so ~30 MB is transferred per tile.
  ALOS    : AW3D30 GeoTIFFs mirrored by OpenTopography
  NASADEM : NASADEM_be GeoTIFFs mirrored by OpenTopography

Usage:
  python dem_elevations.py "input.csv" [-o out.csv] [--cache-dir dem_cache]
                           [-j 8] [--interpolation bilinear|nearest] [--limit N]
"""

import argparse
import csv
import functools
import math
import multiprocessing
import os
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

DEM_NAMES = ["GLO-30", "FABDEM", "ALOS", "NASADEM"]

GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
ALOS_BASE = "https://opentopography.s3.sdsc.edu/raster/AW3D30/AW3D30_global"
NASADEM_BASE = "https://opentopography.s3.sdsc.edu/raster/NASADEM/NASADEM_be"
FABDEM_BASE = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"

# Geoid undulation grids (band value N; ellipsoidal = orthometric + N)
GEOID_URLS = {
    "egm96": "https://cdn.proj.org/us_nga_egm96_15.tif",
    "egm2008": "https://cdn.proj.org/us_nga_egm08_25.tif",
}
DEM_GEOID = {"GLO-30": "egm2008", "FABDEM": "egm2008",
             "ALOS": "egm96", "NASADEM": "egm96"}

# GLO-30 auxiliary masks (per-pixel provenance), decoded per the Copernicus
# DEM Product Handbook. HEM is the interferometric height error in meters --
# note it reflects relative precision only and is blind to absolute errors
# such as phase-unwrapping busts on isolated plateaus.
AUX_KINDS = ("EDM", "FLM", "HEM")
EDM_LEGEND = {
    0: "void", 1: "not edited", 2: "infill of external DEM",
    3: "interpolated", 4: "smoothed", 5: "airport editing",
    6: "raised negative elevation", 7: "flattened", 8: "ocean", 9: "lake",
    10: "river", 11: "shoreline", 12: "morphological editing", 13: "shifted",
}
FLM_LEGEND = {
    0: "void", 1: "edited (see EDM)", 2: "not edited/filled", 3: "ASTER",
    4: "SRTM90", 5: "SRTM30", 6: "GMTED2010", 7: "SRTM30plus",
    8: "TerraSAR-X radargrammetric DEM", 9: "AW3D30", 10: "national DEM",
}
MASK_HEADERS = ["GLO-30 EDM", "GLO-30 FLM", "GLO-30 HEM (m)"]


# ---------------------------------------------------------------- tile naming

def _lat_label(v, width=2):
    return ("N" if v >= 0 else "S") + f"{abs(v):0{width}d}"


def _lon_label(v, width=3):
    return ("E" if v >= 0 else "W") + f"{abs(v):0{width}d}"


def tile_origin(lat, lon):
    """SW corner of the 1x1 degree tile containing the point."""
    return math.floor(lat), math.floor(lon)


def glo30_name(lat0, lon0):
    return f"Copernicus_DSM_COG_10_{_lat_label(lat0)}_00_{_lon_label(lon0)}_00_DEM"


def alos_name(lat0, lon0):
    return f"ALPSMLC30_{_lat_label(lat0, 3)}{_lon_label(lon0)}_DSM.tif"


def nasadem_name(lat0, lon0):
    return f"NASADEM_HGT_{_lat_label(lat0).lower()}{_lon_label(lon0).lower()}.tif"


def fabdem_tile_name(lat0, lon0):
    return f"{_lat_label(lat0)}{_lon_label(lon0)}_FABDEM_V1-2.tif"


def fabdem_zip_name(lat0, lon0):
    """10x10 degree block zip containing the 1-degree tile."""
    blat = math.floor(lat0 / 10) * 10
    blon = math.floor(lon0 / 10) * 10
    return (f"{_lat_label(blat)}{_lon_label(blon)}-"
            f"{_lat_label(blat + 10)}{_lon_label(blon + 10)}_FABDEM_V1-2.zip")


# ---------------------------------------------------------------- downloading

_session = None


def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(total=4, backoff_factor=2,
                        status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session


def download(url, dest):
    """Download url to dest atomically. Returns dest, or None on HTTP 404.

    A 404 is remembered with a <dest>.missing marker so it is not retried
    on later runs (ocean blocks have no tile).
    """
    dest = Path(dest)
    missing = dest.with_suffix(dest.suffix + ".missing")
    if dest.exists():
        return dest
    if missing.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get_session().get(url, stream=True, timeout=(30, 300))
    if r.status_code == 404:
        missing.touch()
        return None
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        os.replace(tmp, dest)  # atomic; safe with concurrent workers
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def ensure_glo30(cache, lat0, lon0):
    name = glo30_name(lat0, lon0)
    return download(f"{GLO30_BASE}/{name}/{name}.tif", cache / "glo30" / f"{name}.tif")


def ensure_alos(cache, lat0, lon0):
    name = alos_name(lat0, lon0)
    return download(f"{ALOS_BASE}/{name}", cache / "alos" / name)


def ensure_nasadem(cache, lat0, lon0):
    name = nasadem_name(lat0, lon0)
    return download(f"{NASADEM_BASE}/{name}", cache / "nasadem" / name)


def ensure_fabdem(cache, lat0, lon0):
    """Get a FABDEM 1-degree tile.

    FABDEM ships as 10x10 degree zips of 1+ GB; rather than download the whole
    zip, extract just the needed tile with HTTP range requests (remotezip).
    """
    from remotezip import RemoteZip
    tif = cache / "fabdem" / fabdem_tile_name(lat0, lon0)
    missing = tif.with_suffix(tif.suffix + ".missing")
    if tif.exists():
        return tif
    if missing.exists():
        return None
    tif.parent.mkdir(parents=True, exist_ok=True)
    url = f"{FABDEM_BASE}/{fabdem_zip_name(lat0, lon0)}"
    if get_session().head(url, timeout=30).status_code == 404:
        missing.touch()  # whole block is ocean
        return None
    with RemoteZip(url) as zf:
        if tif.name not in zf.namelist():
            missing.touch()  # tile absent inside the block (ocean)
            return None
        tmp = tif.with_suffix(tif.suffix + f".part{os.getpid()}")
        try:
            with zf.open(tif.name) as src, open(tmp, "wb") as out:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
            os.replace(tmp, tif)
        finally:
            tmp.unlink(missing_ok=True)
    return tif


ENSURE = {"GLO-30": ensure_glo30, "FABDEM": ensure_fabdem,
          "ALOS": ensure_alos, "NASADEM": ensure_nasadem}


def ensure_geoid(cache, model):
    url = GEOID_URLS[model]
    return download(url, cache / "geoid" / url.rsplit("/", 1)[1])


def _ensure_glo30_aux(kind, cache, lat0, lon0):
    name = glo30_name(lat0, lon0)          # ..._DEM
    stem = name[: -len("_DEM")]
    # NB: not "aux" -- that is a reserved device name on Windows
    return download(f"{GLO30_BASE}/{name}/AUXFILES/{stem}_{kind}.tif",
                    cache / "glo30" / "masks" / f"{stem}_{kind}.tif")


ENSURE.update({k: functools.partial(_ensure_glo30_aux, k) for k in AUX_KINDS})


# ------------------------------------------------------- parallel tile fetch

_worker_cache = None


def _init_worker(cache_dir):
    global _worker_cache
    _worker_cache = Path(cache_dir)


def _fetch_task(task):
    dem, lat0, lon0 = task
    try:
        path = ENSURE[dem](_worker_cache, lat0, lon0)
        return task, "cached" if path else "no tile (ocean?)", None
    except Exception as e:
        return task, None, f"{type(e).__name__}: {e}"


def fetch_all_tiles(cache, points, workers, extra=()):
    """Download every unique tile needed by `points` (list of (lat, lon))."""
    layers = list(DEM_NAMES) + list(extra)
    tasks = sorted({(dem, *tile_origin(lat, lon))
                    for lat, lon in points for dem in layers})
    print(f"{len(tasks)} unique tiles needed for {len(points)} points")
    failures = []

    def report(i, result):
        (dem, lat0, lon0), status, err = result
        label = f"[{i}/{len(tasks)}] {dem} {_lat_label(lat0)}{_lon_label(lon0)}"
        if err:
            failures.append(result)
            print(f"{label}: ERROR {err}", flush=True)
        else:
            print(f"{label}: {status}", flush=True)

    if workers > 1:
        with multiprocessing.Pool(workers, _init_worker, (str(cache),)) as pool:
            for i, result in enumerate(pool.imap(_fetch_task, tasks), 1):
                report(i, result)
    else:
        _init_worker(str(cache))
        for i, task in enumerate(tasks, 1):
            report(i, _fetch_task(task))
    return failures


# ------------------------------------------------------------------ sampling

def sample(dataset, lat, lon, interpolation):
    """Elevation at (lat, lon) from an open rasterio dataset, or None."""
    import numpy as np
    colf, rowf = ~dataset.transform * (lon, lat)
    fx, fy = colf - 0.5, rowf - 0.5  # pixel-center coordinates
    x0, y0 = math.floor(fx), math.floor(fy)
    win = ((y0, y0 + 2), (x0, x0 + 2))
    block = dataset.read(1, window=win, boundless=True, masked=True)
    if interpolation == "bilinear" and not block.mask.any():
        wx, wy = fx - x0, fy - y0
        top = block[0, 0] * (1 - wx) + block[0, 1] * wx
        bot = block[1, 0] * (1 - wx) + block[1, 1] * wx
        val = top * (1 - wy) + bot * wy
    else:  # nearest (also the fallback when a neighbor is nodata)
        val = block[int(round(fy)) - y0, int(round(fx)) - x0]
    if val is np.ma.masked or (hasattr(val, "item") and not np.isfinite(val)):
        return None
    return float(val)


class TileSampler:
    def __init__(self, cache, interpolation, geoid=True):
        self.cache = cache
        self.interpolation = interpolation
        self.geoid = geoid
        self.datasets = {}  # path -> open dataset

    def _sample_path(self, path, lat, lon, interpolation):
        import rasterio
        key = str(path)
        if key not in self.datasets:
            self.datasets[key] = rasterio.open(path)
        return sample(self.datasets[key], lat, lon, interpolation)

    def undulation(self, dem, lat, lon):
        """Geoid height N for this DEM's vertical datum (always bilinear —
        the grids are coarse, 2.5-15 arcmin)."""
        path = ensure_geoid(self.cache, DEM_GEOID[dem])
        if path is None:
            raise RuntimeError(f"geoid grid unavailable: {DEM_GEOID[dem]}")
        return self._sample_path(path, lat, lon, "bilinear")

    def elevation(self, dem, lat, lon):
        path = ENSURE[dem](self.cache, *tile_origin(lat, lon))
        if path is None:
            return None
        val = self._sample_path(path, lat, lon, self.interpolation)
        if val is not None and self.geoid:
            val += self.undulation(dem, lat, lon)
        return val

    def mask_info(self, lat, lon):
        """GLO-30 provenance at the point: [EDM label, FLM label, HEM str]."""
        out = []
        for kind in AUX_KINDS:
            path = ENSURE[kind](self.cache, *tile_origin(lat, lon))
            v = (self._sample_path(path, lat, lon, "nearest")
                 if path is not None else None)
            if kind == "HEM":
                # HEM nodata is -32767 (may not be declared in the tif)
                out.append("" if v is None or v < -32000 else f"{v:.2f}")
            else:
                legend = EDM_LEGEND if kind == "EDM" else FLM_LEGEND
                out.append("" if v is None else legend.get(int(v), str(int(v))))
        return out

    def close(self):
        for ds in self.datasets.values():
            ds.close()


# ----------------------------------------------------------------------- CSV

def find_header(rows):
    """Return (index, {colname_lower: colindex}) of the row with Lat/Long."""
    for i, row in enumerate(rows):
        cols = {c.strip().lower(): j for j, c in enumerate(row) if c.strip()}
        if "lat" in cols and "long" in cols:
            return i, cols
    sys.exit("error: no header row containing 'Lat' and 'Long' columns found")


def parse_point(row, lat_col, lon_col):
    try:
        return float(row[lat_col]), float(row[lon_col])
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="input CSV with Lat/Long columns (WGS84)")
    ap.add_argument("-o", "--output",
                    help="output CSV (default: <input>_dem_elevations.csv)")
    ap.add_argument("--cache-dir", default="dem_cache",
                    help="tile cache directory (default: dem_cache)")
    ap.add_argument("-j", "--workers", type=int, default=1,
                    help="parallel download processes (default: 1 = serial)")
    ap.add_argument("--interpolation", choices=["bilinear", "nearest"],
                    default="nearest", help="sampling method (default: nearest)")
    ap.add_argument("--no-geoid", action="store_true",
                    help="report raw orthometric DEM values instead of adding "
                         "the geoid undulation to get ellipsoidal heights")
    ap.add_argument("--no-masks", action="store_true",
                    help="skip the GLO-30 EDM/FLM/HEM provenance columns")
    ap.add_argument("--limit", type=int,
                    help="only process the first N data rows (for testing)")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out_path = args.output or str(Path(args.csv).with_suffix("")) + "_dem_elevations.csv"

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = [list(r) for r in csv.reader(f)]
    hdr_i, cols = find_header(rows)
    header = rows[hdr_i]
    lat_col, lon_col = cols["lat"], cols["long"]
    dem_cols = {}
    for dem in DEM_NAMES:
        if dem.lower() in cols:
            dem_cols[dem] = cols[dem.lower()]
        else:
            header.append(dem)
            dem_cols[dem] = len(header) - 1
    mask_cols = []
    if not args.no_masks:
        for h in MASK_HEADERS:
            if h.lower() in cols:
                mask_cols.append(cols[h.lower()])
            else:
                header.append(h)
                mask_cols.append(len(header) - 1)

    data = []  # (row_index, lat, lon)
    for i in range(hdr_i + 1, len(rows)):
        pt = parse_point(rows[i], lat_col, lon_col)
        if pt:
            data.append((i, *pt))
    if args.limit:
        data = data[:args.limit]
    print(f"{len(data)} points to process")

    failures = fetch_all_tiles(cache, [(la, lo) for _, la, lo in data],
                               args.workers,
                               extra=() if args.no_masks else AUX_KINDS)
    if failures:
        print(f"warning: {len(failures)} tile downloads failed; "
              "their points will be left blank", file=sys.stderr)

    if not args.no_geoid:
        for model in sorted(set(DEM_GEOID.values())):
            print(f"fetching geoid grid: {GEOID_URLS[model].rsplit('/', 1)[1]}",
                  flush=True)
            if ensure_geoid(cache, model) is None:
                sys.exit(f"error: geoid grid {model} not found on cdn.proj.org")

    sampler = TileSampler(cache, args.interpolation, geoid=not args.no_geoid)
    try:
        for n, (i, lat, lon) in enumerate(data, 1):
            row = rows[i]
            for dem in DEM_NAMES:
                col = dem_cols[dem]
                while len(row) <= col:
                    row.append("")
                try:
                    elev = sampler.elevation(dem, lat, lon)
                except Exception as e:
                    print(f"row {i + 1} {dem}: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    elev = ""
                row[col] = f"{elev:.2f}" if isinstance(elev, float) else ""
            if mask_cols:
                try:
                    values = sampler.mask_info(lat, lon)
                except Exception as e:
                    print(f"row {i + 1} masks: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    values = ["", "", ""]
                for col, v in zip(mask_cols, values):
                    while len(row) <= col:
                        row.append("")
                    row[col] = v
            if n % 50 == 0 or n == len(data):
                print(f"sampled {n}/{len(data)}", flush=True)
    finally:
        sampler.close()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
