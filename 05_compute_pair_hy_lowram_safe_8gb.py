#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, NamedTuple

# Keep BLAS and allocator fan-out conservative when this file is run directly on
# a shared 8GB server. Explicit user settings still take precedence.
for _env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_name, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import duckdb
import numpy as np

from common import dt_to_ns, instrument_key, ns_to_dt, parse_dt, sql_quote


EMPTY_I64 = np.array([], dtype=np.int64)
EMPTY_F64 = np.array([], dtype=np.float64)
SCRIPT_VERSION = "3.3.0"
ESTIMATED_CSV_BYTES_PER_ROW = 600
CHUNK_RE = re.compile(
    r"chunk_start=(\d{8}T\d{6})_chunk_end=(\d{8}T\d{6})"
)


class CacheIntegrityError(RuntimeError):
    """The interval cache cannot safely support the requested calculation."""


class CacheLocation(NamedTuple):
    """One resolved cache layout for one instrument and price mode."""

    root: Path
    layout: str
    config: dict | None
    max_interval_ns: int
    date_partitioned: bool


_CACHE_LOCATIONS: dict[tuple[str, str, str, str, str], CacheLocation] = {}
_NON_DATE_FILES: dict[str, list[str]] = {}
_CACHE_EXCLUDED_RANGES: dict[str, tuple[tuple[int, int], ...]] = {}
_WARNED: set[str] = set()


class YWindowState(NamedTuple):
    """Lag-independent Y components for one accounting window."""

    var_y: float
    n_y: int
    cov_first: int
    cov_stop: int


class PairAccumulator(NamedTuple):
    sum_cov: np.ndarray
    sum_var_x: np.ndarray
    sum_var_y: np.ndarray
    total_overlap: np.ndarray
    total_n_x: np.ndarray
    total_n_y: np.ndarray
    n_windows: np.ndarray


def sql_list(paths: list[str]) -> str:
    """DuckDB SQL list literal for read_parquet([...])."""
    return "[" + ", ".join(sql_quote(p) for p in paths) + "]"


def parquet_scan(files: list[str]) -> str:
    return f"read_parquet({sql_list(files)}, union_by_name=true, hive_partitioning=true)"


def scan_columns(con: duckdb.DuckDBPyConnection, files: list[str]) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {parquet_scan(files)}").fetchall()
    return {r[0] for r in rows}


def scan_physical_columns(
    con: duckdb.DuckDBPyConnection,
    files: list[str],
) -> set[str]:
    scan = (
        f"read_parquet({sql_list(files)}, union_by_name=true, "
        "hive_partitioning=false)"
    )
    rows = con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    return {r[0] for r in rows}


def warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[WARN] {message}")


def first_glob(pattern: Path) -> str | None:
    return next(glob.iglob(str(pattern)), None)


def load_cache_config(root: Path) -> dict | None:
    path = root / "_cache_config.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"Cannot read cache config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"Cache config is not a JSON object: {path}")
    return value


def cache_excluded_ranges(
    location: CacheLocation,
) -> tuple[tuple[int, int], ...]:
    """Return sorted intentional source gaps recorded by the cache builder."""
    cache_key = str(location.root.resolve())
    cached = _CACHE_EXCLUDED_RANGES.get(cache_key)
    if cached is not None:
        return cached

    config = location.config or {}
    raw_values = config.get("excluded_hours_utc", [])
    if not isinstance(raw_values, list):
        raise CacheIntegrityError(
            f"excluded_hours_utc must be a list in {location.root / '_cache_config.json'}"
        )

    starts: set[int] = set()
    for raw_value in raw_values:
        try:
            hour_start = parse_dt(str(raw_value))
        except (TypeError, ValueError) as exc:
            raise CacheIntegrityError(
                f"Invalid excluded hour {raw_value!r} in "
                f"{location.root / '_cache_config.json'}"
            ) from exc
        if (
            hour_start is None
            or hour_start.minute != 0
            or hour_start.second != 0
            or hour_start.microsecond != 0
        ):
            raise CacheIntegrityError(
                f"Excluded hour is not UTC-hour aligned: {raw_value!r} in "
                f"{location.root / '_cache_config.json'}"
            )
        starts.add(dt_to_ns(hour_start))

    one_hour_ns = 3_600 * 1_000_000_000
    ranges = tuple((start_ns, start_ns + one_hour_ns) for start_ns in sorted(starts))
    _CACHE_EXCLUDED_RANGES[cache_key] = ranges
    return ranges


def ranges_overlap(
    ranges: tuple[tuple[int, int], ...],
    start_ns: int,
    end_ns: int,
) -> bool:
    for excluded_start_ns, excluded_end_ns in ranges:
        if excluded_start_ns >= end_ns:
            return False
        if excluded_end_ns > start_ns:
            return True
    return False


def range_fully_excluded(
    ranges: tuple[tuple[int, int], ...],
    start_ns: int,
    end_ns: int,
) -> bool:
    """Return whether exclusions cover the entire half-open range."""
    if end_ns <= start_ns:
        return True
    cursor = start_ns
    for excluded_start_ns, excluded_end_ns in ranges:
        if excluded_end_ns <= cursor:
            continue
        if excluded_start_ns > cursor:
            return False
        cursor = max(cursor, excluded_end_ns)
        if cursor >= end_ns:
            return True
    return False


def instrument_cache_base(interval_root: str, inst: dict[str, str]) -> Path:
    return (
        Path(interval_root)
        / f"exchange={inst['exchange']}"
        / f"market_type={inst['market_type']}"
        / f"symbol={inst['symbol']}"
    )


def resolve_cache_location(
    interval_root: str,
    inst: dict[str, str],
    price_mode: str,
) -> CacheLocation:
    """Resolve exactly one layout; never combine new and legacy Hive layouts."""
    key = (
        str(Path(interval_root).resolve()),
        inst["exchange"],
        inst["market_type"],
        inst["symbol"],
        price_mode,
    )
    cached = _CACHE_LOCATIONS.get(key)
    if cached is not None:
        return cached

    base = instrument_cache_base(interval_root, inst)
    new_root = base / f"price_mode={price_mode}"

    new_date_file = first_glob(new_root / "date=*" / "*.parquet")
    new_flat_file = first_glob(new_root / "*.parquet")
    new_config = load_cache_config(new_root)
    # An empty directory without a manifest is not enough evidence to shadow a
    # complete legacy cache. A real new-builder run writes the manifest first.
    new_intent = new_config is not None
    new_exists = new_date_file is not None or new_flat_file is not None

    legacy_date_file = first_glob(base / "date=*" / "*.parquet")
    legacy_flat_file = first_glob(base / "*.parquet")
    legacy_config = load_cache_config(base)
    legacy_exists = legacy_date_file is not None or legacy_flat_file is not None

    if new_date_file is not None and new_flat_file is not None:
        raise CacheIntegrityError(
            f"Mixed date-partitioned and flat files under {new_root}; use one layout"
        )
    if legacy_date_file is not None and legacy_flat_file is not None:
        raise CacheIntegrityError(
            f"Mixed date-partitioned and flat legacy files under {base}; use one layout"
        )

    if new_intent or new_exists:
        if legacy_exists:
            warn_once(
                f"mixed-layout:{base}:{price_mode}",
                f"Both new and legacy caches exist under {base}; ignoring legacy "
                f"and using only {new_root}",
            )
        root = new_root
        layout = "partitioned_price_mode"
        config = new_config
        date_partitioned = new_date_file is not None
    elif legacy_exists or legacy_config is not None:
        root = base
        layout = "legacy"
        config = legacy_config
        date_partitioned = legacy_date_file is not None
    else:
        other_modes = sorted(p.name for p in base.glob("price_mode=*") if p.is_dir())
        suffix = f"; available modes={other_modes}" if other_modes else ""
        raise CacheIntegrityError(
            f"No interval cache for {instrument_key(inst)} price_mode={price_mode} "
            f"under {base}{suffix}"
        )

    max_interval_ns = 1_000_000  # 1ms fallback fixes exact-midnight discovery.
    if config is not None and config.get("max_interval_ms") is not None:
        try:
            max_interval_ms = float(config["max_interval_ms"])
        except (TypeError, ValueError) as exc:
            raise CacheIntegrityError(
                f"Invalid max_interval_ms in {root / '_cache_config.json'}"
            ) from exc
        if not math.isfinite(max_interval_ms) or max_interval_ms <= 0:
            raise CacheIntegrityError(
                f"max_interval_ms must be finite and > 0 in {root / '_cache_config.json'}"
            )
        max_interval_ns = max(1, math.ceil(max_interval_ms * 1_000_000))

    location = CacheLocation(
        root=root,
        layout=layout,
        config=config,
        max_interval_ns=max_interval_ns,
        date_partitioned=date_partitioned,
    )
    _CACHE_LOCATIONS[key] = location
    return location


def date_strings_for_ns(start_ns: int, end_ns: int) -> list[str]:
    """Return date=YYYY-MM-DD partitions that can contain rows in [start_ns, end_ns)."""
    if end_ns <= start_ns:
        return []

    d0 = ns_to_dt(start_ns).date()
    # end_ns is exclusive; subtract 1ns so exact midnight does not include next day.
    d1 = ns_to_dt(end_ns - 1).date()

    out: list[str] = []
    d = d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def interval_files_for_range(
    interval_root: str,
    inst: dict[str, str],
    price_mode: str,
    start_ns: int,
    end_ns: int,
    max_files: int | None = None,
    max_end_ns: int | None = None,
) -> list[str]:
    """Return files from one resolved layout, including end-partition lookahead."""
    if end_ns <= start_ns:
        return []
    location = resolve_cache_location(interval_root, inst, price_mode)

    if location.date_partitioned:
        # Cache files are partitioned by interval end time. An interval that starts
        # before read_end may finish later, up to the manifest's max interval cap.
        partition_end_ns = end_ns + location.max_interval_ns
        if max_end_ns is not None:
            # Files are partitioned by interval end. When the estimator has a
            # hard observation-horizon cap, no partition after that end can
            # contribute even if an interval starts inside the wider read range.
            partition_end_ns = min(partition_end_ns, max_end_ns + 1)
        files: list[str] = []
        for date_s in date_strings_for_ns(start_ns, partition_end_ns):
            for file in glob.iglob(
                str(location.root / f"date={date_s}" / "*.parquet")
            ):
                files.append(file)
                if max_files is not None and len(files) > max_files:
                    raise CacheIntegrityError(
                        f"More than {max_files:,} cache files are needed for "
                        f"{instrument_key(inst)} in one requested range. This is "
                        "unsafe on an 8GB server; compact small files or reduce "
                        "the date shard."
                    )
        return sorted(set(files))

    cache_key = str(location.root.resolve())
    if cache_key not in _NON_DATE_FILES:
        files = []
        for path in location.root.rglob("*.parquet"):
            if not path.is_file():
                continue
            files.append(str(path))
            if max_files is not None and len(files) > max_files:
                raise CacheIntegrityError(
                    f"More than {max_files:,} cache files were found under "
                    f"non-date cache {location.root}. Compact or rebuild it as "
                    "date-partitioned data before using an 8GB server."
                )
        _NON_DATE_FILES[cache_key] = sorted(files)
    elif max_files is not None and len(_NON_DATE_FILES[cache_key]) > max_files:
        raise CacheIntegrityError(
            f"Cached file list for {location.root} exceeds {max_files:,} files"
        )
    return _NON_DATE_FILES[cache_key]


def handle_missing_cache(
    policy: str,
    inst: dict[str, str],
    price_mode: str,
    start_ns: int,
    end_ns: int,
) -> None:
    message = (
        f"No cache files for {instrument_key(inst)} price_mode={price_mode} "
        f"in UTC range [{ns_to_dt(start_ns)}, {ns_to_dt(end_ns)})"
    )
    if policy == "error":
        raise CacheIntegrityError(message)
    if policy == "warn":
        warn_once(
            f"missing:{instrument_key(inst)}:{price_mode}",
            message + "; treating this and later missing ranges as empty",
        )


def load_intervals(
    con: duckdb.DuckDBPyConnection,
    interval_root: str,
    inst: dict[str, str],
    start_ns: int,
    end_ns: int,
    price_mode: str,
    *,
    missing_cache_policy: str = "error",
    allow_unlabeled_legacy: bool = False,
    max_rows: int | None = None,
    max_cache_files: int | None = None,
    max_end_ns: int | None = None,
):
    location = resolve_cache_location(interval_root, inst, price_mode)
    files = interval_files_for_range(
        interval_root,
        inst,
        price_mode,
        start_ns,
        end_ns,
        max_files=max_cache_files,
        max_end_ns=max_end_ns,
    )
    if not files:
        handle_missing_cache(
            missing_cache_policy, inst, price_mode, start_ns, end_ns
        )
        return EMPTY_I64, EMPTY_I64, EMPTY_F64

    price_filter = ""
    cols = scan_columns(con, files)
    required = {"start_ns", "end_ns", "ret"}
    if not required.issubset(cols):
        raise CacheIntegrityError(
            f"Cache schema missing {sorted(required - cols)} for {instrument_key(inst)}"
        )

    if location.layout == "legacy":
        physical_cols = scan_physical_columns(con, files)
        if "price_mode" not in physical_cols:
            config = location.config or {}
            manifest_labels_mode = (
                config.get("cache_config_version") == 2
                and config.get("exchange") == inst["exchange"]
                and config.get("market_type") == inst["market_type"]
                and config.get("symbol") == inst["symbol"]
                and config.get("price_mode") == price_mode
            )
            if manifest_labels_mode:
                pass
            elif not allow_unlabeled_legacy:
                raise CacheIntegrityError(
                    f"Legacy cache {location.root} has no physical price_mode column; "
                    "refusing to label it from the command line without a trusted "
                    "manifest. Rebuild with the price_mode=... layout or pass "
                    "--allow-unlabeled-legacy-price-mode."
                )
            else:
                warn_once(
                    f"unlabeled-legacy:{location.root}",
                    f"Trusting unlabeled legacy cache {location.root} as {price_mode}",
                )
        else:
            price_filter = (
                f"AND CAST(price_mode AS VARCHAR) = {sql_quote(price_mode)}"
            )
    elif "price_mode" in cols:
        price_filter = f"AND CAST(price_mode AS VARCHAR) = {sql_quote(price_mode)}"

    max_end_filter = (
        "" if max_end_ns is None else f"AND end_ns <= {max_end_ns}"
    )
    where_sql = f"""
        end_ns > {start_ns}
        AND start_ns < {end_ns}
        AND start_ns IS NOT NULL
        AND end_ns IS NOT NULL
        AND ret IS NOT NULL
        AND end_ns > start_ns
        AND isfinite(CAST(ret AS DOUBLE))
        {max_end_filter}
        {price_filter}
    """
    scan = parquet_scan(files)

    if max_rows is not None:
        row_count = int(
            con.execute(f"""
                SELECT count(*)
                FROM (
                    SELECT 1
                    FROM {scan}
                    WHERE {where_sql}
                    LIMIT {max_rows + 1}
                )
            """).fetchone()[0]
        )
        if row_count == 0:
            return EMPTY_I64, EMPTY_I64, EMPTY_F64
        if row_count > max_rows:
            required_mib = row_count * 24 / (1024 ** 2)
            allowed_mib = max_rows * 24 / (1024 ** 2)
            raise RuntimeError(
                f"Window preflight for {instrument_key(inst)} found "
                f"{row_count:,} rows (~{required_mib:.1f}MiB base NumPy), exceeding "
                f"the remaining resource allowance {max_rows:,} rows "
                f"(~{allowed_mib:.1f}MiB). Reduce --window-hours, lag count, or "
                "raise the relevant guard only after checking server limits."
            )

    result = con.execute(f"""
        SELECT
            CAST(start_ns AS BIGINT) AS start_ns,
            CAST(end_ns AS BIGINT) AS end_ns,
            CAST(ret AS DOUBLE) AS ret
        FROM {scan}
        WHERE {where_sql}
        ORDER BY end_ns
    """).fetchnumpy()

    if len(result["ret"]) == 0:
        return EMPTY_I64, EMPTY_I64, EMPTY_F64

    start_arr = np.asarray(result["start_ns"], dtype=np.int64)
    end_arr = np.asarray(result["end_ns"], dtype=np.int64)
    ret_arr = np.asarray(result["ret"], dtype=np.float64)

    if not np.all(np.isfinite(ret_arr)):
        raise CacheIntegrityError(
            f"Non-finite return escaped cache filtering for {instrument_key(inst)}"
        )

    # ORDER BY guarantees nondecreasing end_ns. The low-copy HY sweep also
    # requires nondecreasing start_ns, which is true for consecutive return
    # intervals. Fail loudly if a malformed cache violates that contract;
    # silently continuing would produce incorrect overlap counts/covariance.
    if len(end_arr) > 1 and np.any(end_arr[1:] < end_arr[:-1]):
        raise RuntimeError(
            f"Interval cache is not sorted by end_ns for {instrument_key(inst)}"
        )
    if len(start_arr) > 1 and np.any(start_arr[1:] < start_arr[:-1]):
        raise RuntimeError(
            "Interval start_ns is not nondecreasing after ORDER BY end_ns for "
            f"{instrument_key(inst)}; the two-pointer HY sweep would be unsafe"
        )
    if len(start_arr) > 1 and np.any(start_arr[1:] < end_arr[:-1]):
        raise CacheIntegrityError(
            f"Overlapping intervals within {instrument_key(inst)} violate the "
            "return-cache contract and can make the HY sweep quadratic"
        )

    return start_arr, end_arr, ret_arr


def validation_issue(policy: str, message: str) -> None:
    if policy == "strict":
        raise CacheIntegrityError(message)
    if policy == "warn":
        warn_once(f"validation:{message}", message)


def parse_config_ns(config: dict, field: str, manifest: Path) -> int:
    try:
        value = parse_dt(str(config.get(field) or ""))
    except (TypeError, ValueError) as exc:
        raise CacheIntegrityError(f"Invalid {field!r} in {manifest}") from exc
    if value is None:
        raise CacheIntegrityError(f"Invalid {field!r} in {manifest}")
    return dt_to_ns(value)


def parse_chunk_span(path: Path) -> tuple[int, int] | None:
    match = CHUNK_RE.search(path.name)
    if match is None:
        return None
    try:
        start = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        end = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    start_ns = dt_to_ns(start)
    end_ns = dt_to_ns(end)
    return (start_ns, end_ns) if start_ns < end_ns else None


def validate_chunk_coverage(
    location: CacheLocation,
    files: list[str],
    required_start_ns: int,
    required_end_ns: int,
    policy: str,
    excluded_ranges: tuple[tuple[int, int], ...] = (),
) -> None:
    """Validate file-level time coverage before any output file is opened."""
    span_counts: dict[tuple[int, int], int] = {}
    unparseable: list[str] = []
    for file in files:
        span = parse_chunk_span(Path(file))
        if span is None:
            unparseable.append(file)
            continue
        if span[1] > required_start_ns and span[0] < required_end_ns:
            span_counts[span] = span_counts.get(span, 0) + 1

    if not files and range_fully_excluded(
        excluded_ranges, required_start_ns, required_end_ns
    ):
        return
    if not files:
        validation_issue(policy, f"No Parquet files under cache root {location.root}")
        return

    if unparseable:
        validation_issue(
            policy,
            f"Cannot verify chunk coverage for {len(unparseable)} candidate files; "
            f"example filename lacks chunk_start/chunk_end: {unparseable[0]}",
        )

    duplicates = [span for span, count in span_counts.items() if count > 1]
    if duplicates:
        example = duplicates[0]
        validation_issue(
            policy,
            f"Duplicate cache files cover the same chunk under {location.root}: "
            f"{ns_to_dt(example[0])} -> {ns_to_dt(example[1])}",
        )

    if not span_counts and range_fully_excluded(
        excluded_ranges, required_start_ns, required_end_ns
    ):
        return
    if not span_counts:
        validation_issue(
            policy,
            f"No cache chunk spans intersect required range "
            f"[{ns_to_dt(required_start_ns)}, {ns_to_dt(required_end_ns)}) "
            f"under {location.root}",
        )
        return

    cursor = required_start_ns
    started = False
    for chunk_start_ns, chunk_end_ns in sorted(span_counts):
        if chunk_end_ns <= cursor:
            continue
        if not started:
            if chunk_start_ns > cursor:
                if not range_fully_excluded(
                    excluded_ranges, cursor, chunk_start_ns
                ):
                    validation_issue(
                        policy,
                        f"Cache chunk gap for {location.root}: {ns_to_dt(cursor)} -> "
                        f"{ns_to_dt(chunk_start_ns)}",
                    )
                    return
            cursor = chunk_end_ns
            started = True
        elif chunk_start_ns < cursor:
            validation_issue(
                policy,
                f"Overlapping cache chunks under {location.root}: next chunk starts "
                f"at {ns_to_dt(chunk_start_ns)} before previous coverage ends at "
                f"{ns_to_dt(cursor)}",
            )
            return
        elif chunk_start_ns > cursor:
            if not range_fully_excluded(
                excluded_ranges, cursor, chunk_start_ns
            ):
                validation_issue(
                    policy,
                    f"Cache chunk gap for {location.root}: {ns_to_dt(cursor)} -> "
                    f"{ns_to_dt(chunk_start_ns)}",
                )
                return
            cursor = chunk_end_ns
        else:
            cursor = chunk_end_ns
        if cursor >= required_end_ns:
            return

    if cursor < required_end_ns and not range_fully_excluded(
        excluded_ranges, cursor, required_end_ns
    ):
        validation_issue(
            policy,
            f"Cache coverage ends at {ns_to_dt(cursor)} but calculation requires "
            f"data-file coverage through {ns_to_dt(required_end_ns)} under "
            f"{location.root}",
        )


def validate_candidate_schemas(
    con: duckdb.DuckDBPyConnection,
    files: list[str],
    required_cols: set[str],
    policy: str,
) -> tuple[bool, bool]:
    """Read every candidate footer so union_by_name cannot hide a bad chunk."""
    if not files:
        return False, False

    any_price_mode = False
    all_price_mode = True
    batch_size = 1_000
    for offset in range(0, len(files), batch_size):
        batch = files[offset : offset + batch_size]
        rows = con.execute(
            f"SELECT file_name, name FROM parquet_schema({sql_list(batch)})"
        ).fetchall()
        schemas: dict[str, set[str]] = {file: set() for file in batch}
        for file_name, column_name in rows:
            schemas.setdefault(str(file_name), set()).add(str(column_name))

        for file in batch:
            columns = schemas.get(file, set())
            missing = required_cols - columns
            if missing:
                validation_issue(
                    policy,
                    f"Cache file {file} is unreadable or missing columns "
                    f"{sorted(missing)}",
                )
            has_mode = "price_mode" in columns
            any_price_mode = any_price_mode or has_mode
            all_price_mode = all_price_mode and has_mode
    return all_price_mode, any_price_mode


def validate_cache_location(
    con: duckdb.DuckDBPyConnection,
    interval_root: str,
    inst: dict[str, str],
    price_mode: str,
    required_start_ns: int,
    required_end_ns: int,
    *,
    policy: str,
    allow_unlabeled_legacy: bool,
    max_cache_files: int | None = None,
) -> None:
    """Validate the role-aware interval-end range needed by the estimator."""
    location = resolve_cache_location(interval_root, inst, price_mode)
    manifest = location.root / "_cache_config.json"

    if not location.date_partitioned:
        validation_issue(
            policy,
            f"Non-date-partitioned cache {location.root} is not accepted by strict "
            "low-RAM validation; rebuild using date=YYYY-MM-DD partitions",
        )

    config = location.config
    manifest_identity_valid = False
    excluded_ranges: tuple[tuple[int, int], ...] = ()
    if config is None:
        validation_issue(policy, f"Missing cache config manifest: {manifest}")
    else:
        try:
            excluded_ranges = cache_excluded_ranges(location)
        except CacheIntegrityError as exc:
            validation_issue(policy, str(exc))
        if config.get("max_interval_ms") is None:
            validation_issue(
                policy,
                f"Missing max_interval_ms in {manifest}; a safe partition "
                "lookahead cannot be inferred",
            )
        if config.get("cache_config_version") != 2:
            validation_issue(
                policy,
                f"Unsupported cache_config_version={config.get('cache_config_version')!r} "
                f"in {manifest}; expected 2",
            )
        expected = {
            "exchange": inst["exchange"],
            "market_type": inst["market_type"],
            "symbol": inst["symbol"],
            "price_mode": price_mode,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            validation_issue(
                policy,
                f"Cache config identity mismatch in {manifest}: {mismatches}",
            )
        else:
            manifest_identity_valid = config.get("cache_config_version") == 2

        try:
            config_start_ns = parse_config_ns(config, "start", manifest)
            config_end_ns = parse_config_ns(config, "end", manifest)
        except CacheIntegrityError as exc:
            validation_issue(policy, str(exc))
        else:
            if config_start_ns > required_start_ns or config_end_ns < required_end_ns:
                validation_issue(
                    policy,
                    f"Cache range [{ns_to_dt(config_start_ns)}, "
                    f"{ns_to_dt(config_end_ns)}) does not cover required estimator "
                    f"end range [{ns_to_dt(required_start_ns)}, "
                    f"{ns_to_dt(required_end_ns)}) for {instrument_key(inst)}",
                )

        input_mode = config.get("input_mode")
        if input_mode == "streaming_sorted_unique":
            warn_once(
                f"streaming-input:{location.root}",
                f"Cache {location.root} trusts that source timestamps were strictly "
                "increasing and unique; the compute step cannot verify that source assumption",
            )

        try:
            chunk_minutes = float(config.get("chunk_minutes"))
        except (TypeError, ValueError):
            chunk_minutes = float("nan")
        if not math.isfinite(chunk_minutes) or chunk_minutes <= 0:
            validation_issue(
                policy,
                f"chunk_minutes must be finite and > 0 in {manifest}",
            )

    candidate_files = interval_files_for_range(
        interval_root,
        inst,
        price_mode,
        required_start_ns,
        required_end_ns,
        max_files=max_cache_files,
        max_end_ns=required_end_ns,
    )
    validate_chunk_coverage(
        location,
        candidate_files,
        required_start_ns,
        required_end_ns,
        policy,
        excluded_ranges,
    )
    if not candidate_files:
        return

    required_cols = {"start_ns", "end_ns", "ret"}
    all_physical_mode, any_physical_mode = validate_candidate_schemas(
        con,
        candidate_files,
        required_cols,
        policy,
    )

    if location.layout == "legacy":
        if any_physical_mode and not all_physical_mode:
            validation_issue(
                policy,
                f"Only some legacy files under {location.root} contain price_mode",
            )
        elif all_physical_mode:
            physical_scan = (
                f"read_parquet({sql_list(candidate_files)}, union_by_name=true, "
                "hive_partitioning=false)"
            )
            nulls, distinct_modes, min_mode, max_mode = con.execute(f"""
                SELECT
                    count(*) FILTER (WHERE price_mode IS NULL),
                    count(DISTINCT CAST(price_mode AS VARCHAR)),
                    min(CAST(price_mode AS VARCHAR)),
                    max(CAST(price_mode AS VARCHAR))
                FROM {physical_scan}
            """).fetchone()
            empty_but_manifest_labeled = (
                distinct_modes == 0 and manifest_identity_valid
            )
            if not empty_but_manifest_labeled and (
                nulls
                or distinct_modes != 1
                or min_mode != price_mode
                or max_mode != price_mode
            ):
                validation_issue(
                    policy,
                    f"Legacy physical price_mode values under {location.root} do "
                    f"not uniquely equal {price_mode!r}: nulls={nulls}, "
                    f"distinct={distinct_modes}, min={min_mode!r}, max={max_mode!r}",
                )
        elif not manifest_identity_valid and not allow_unlabeled_legacy:
            validation_issue(
                policy,
                f"Legacy cache {location.root} has neither a trusted manifest nor "
                "a physical price_mode column",
            )


def arrays_nbytes(*arrays: np.ndarray) -> int:
    return sum(int(array.nbytes) for array in arrays)


def corr_or_none(cov: float, var_x: float, var_y: float) -> float | None:
    """Return a nullable correlation without multiplying variances first."""
    if not all(math.isfinite(value) for value in (cov, var_x, var_y)):
        return None
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    value = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return value if math.isfinite(value) else None


def stable_config_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_identity(location: CacheLocation) -> dict[str, object]:
    config = location.config
    return {
        "layout": location.layout,
        "max_interval_ms": location.max_interval_ns / 1_000_000.0,
        "cache_config_version": (
            config.get("cache_config_version") if config is not None else None
        ),
        "drop_zero_returns": (
            config.get("drop_zero_returns") if config is not None else None
        ),
        "cache_config_hash": (
            stable_config_hash(config) if config is not None else None
        ),
    }


def ensure_free_disk(path: Path, min_free_gb: float, label: str) -> None:
    if min_free_gb <= 0:
        return
    free = shutil.disk_usage(path).free
    required = int(min_free_gb * 1024 ** 3)
    if free < required:
        raise RuntimeError(
            f"Only {free / 1024 ** 3:.1f}GiB free on {label} filesystem at "
            f"{path}; require at least {min_free_gb:.1f}GiB"
        )


def parse_size_bytes(value: str) -> int:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Cannot estimate size {value!r}; use a value such as 8GB or 4096MiB"
        )
    amount = float(match.group(1))
    unit = match.group(2).lower()
    powers = {
        "b": 0,
        "kb": 1,
        "kib": 1,
        "mb": 2,
        "mib": 2,
        "gb": 3,
        "gib": 3,
        "tb": 4,
        "tib": 4,
    }
    return math.ceil(amount * (1024 ** powers[unit]))


def ensure_startup_disk_budget(
    output_dir: Path,
    temp_dir: Path,
    min_free_gb: float,
    max_temp_size: str,
    estimated_output_rows: int,
) -> None:
    reserve = int(min_free_gb * 1024 ** 3)
    temp_budget = parse_size_bytes(max_temp_size)
    # CSV and Parquet coexist during conversion. Identity columns intentionally
    # make rows wider, so retain a conservative per-row estimate.
    output_budget = estimated_output_rows * ESTIMATED_CSV_BYTES_PER_ROW * 2
    same_filesystem = output_dir.stat().st_dev == temp_dir.stat().st_dev

    if same_filesystem:
        required = reserve + temp_budget + output_budget
        free = shutil.disk_usage(output_dir).free
        if free < required:
            raise RuntimeError(
                f"Output and temp share a filesystem with {free / 1024 ** 3:.1f}GiB "
                f"free, but the safety budget requires {required / 1024 ** 3:.1f}GiB "
                "(reserve + max spill + estimated CSV/Parquet). Reduce the shard, "
                "lower --max-temp-size, or use a separate temp disk."
            )
        return

    output_free = shutil.disk_usage(output_dir).free
    temp_free = shutil.disk_usage(temp_dir).free
    if output_free < reserve + output_budget:
        raise RuntimeError(
            f"Output filesystem needs {(reserve + output_budget) / 1024 ** 3:.1f}GiB "
            f"free by safety estimate; only {output_free / 1024 ** 3:.1f}GiB available"
        )
    if temp_free < reserve + temp_budget:
        raise RuntimeError(
            f"Temp filesystem needs {(reserve + temp_budget) / 1024 ** 3:.1f}GiB "
            f"free; only {temp_free / 1024 ** 3:.1f}GiB available"
        )


def ensure_parquet_conversion_space(
    output_dir: Path,
    csv_path: Path,
    min_free_gb: float,
) -> None:
    reserve = int(min_free_gb * 1024 ** 3)
    required = reserve + csv_path.stat().st_size
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise RuntimeError(
            f"Need at least {required / 1024 ** 3:.1f}GiB free before CSV-to-Parquet "
            f"conversion; only {free / 1024 ** 3:.1f}GiB available"
        )


def acquire_lock(path: Path, label: str):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{label} lock already exists: {path}. Another task may be running; "
            "if no such process exists, remove the stale lock manually."
        ) from exc
    os.write(fd, f"pid={os.getpid()}\n".encode())
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            os.close(fd)
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    atexit.register(release)
    return release


def variance_by_shifted_end(
    end_ns: np.ndarray,
    ret: np.ndarray,
    shift_ns: int,
    window_start_ns: int,
    window_end_ns: int,
) -> tuple[float, int]:
    """Return sum(ret^2) and count for shifted ends in (window_start, window_end]."""
    first = int(
        np.searchsorted(end_ns, window_start_ns - shift_ns, side="right")
    )
    stop = int(
        np.searchsorted(end_ns, window_end_ns - shift_ns, side="right")
    )
    n = stop - first
    if n <= 0:
        return 0.0, 0

    # A basic slice is a view, unlike boolean indexing, so this does not copy
    # all selected returns. Using the same view twice also avoids the two
    # temporary arrays created by np.dot(ret[mask], ret[mask]).
    selected = ret[first:stop]
    return float(np.dot(selected, selected)), n


def overlap_candidate_bounds(
    start_ns: np.ndarray,
    end_ns: np.ndarray,
    shift_ns: int,
    window_start_ns: int,
    window_end_ns: int,
) -> tuple[int, int]:
    """Contiguous bounds for shifted intervals intersecting the open window."""
    # shifted_end > window_start
    first = int(
        np.searchsorted(end_ns, window_start_ns - shift_ns, side="right")
    )
    # shifted_start < window_end
    stop = int(
        np.searchsorted(start_ns, window_end_ns - shift_ns, side="left")
    )
    if stop < first:
        stop = first
    return first, stop


def prepare_y_window(
    y_start: np.ndarray,
    y_end: np.ndarray,
    y_ret: np.ndarray,
    window_start_ns: int,
    window_end_ns: int,
) -> YWindowState:
    """Compute the Y values that are identical for every X lag in this window."""
    var_y, n_y = variance_by_shifted_end(
        y_end,
        y_ret,
        0,
        window_start_ns,
        window_end_ns,
    )
    cov_first, cov_stop = overlap_candidate_bounds(
        y_start,
        y_end,
        0,
        window_start_ns,
        window_end_ns,
    )
    return YWindowState(var_y, n_y, cov_first, cov_stop)


def hy_for_lag_boundary_fixed(
    x_start: np.ndarray,
    x_end: np.ndarray,
    x_ret: np.ndarray,
    y_start: np.ndarray,
    y_end: np.ndarray,
    y_ret: np.ndarray,
    lag_ns: int,
    window_start_ns: int,
    window_end_ns: int,
    y_window: YWindowState | None = None,
    analysis_end_ns: int | None = None,
    x_variance: tuple[float, int] | None = None,
):
    """
    Same accounting convention as hy_spot_perp_duckdb_boundary_fixed.py.

    Lag convention:
      lag_ms > 0 shifts X intervals forward. If corr peaks at +L, interpret as
      X leading Y by about L ms.

    Window convention:
      * X variance uses the shifted X interval end time.
      * Y variance uses the Y interval end time.
      * HY covariance is assigned by the TRUE overlap end time:
            overlap_end = min(shifted_x_end, y_end)
        This makes adjacent windows summable without double-counting overlap pairs.

    Finite-horizon convention:
      * Production calls pass the whole task's analysis_end_ns.
      * A covariance pair is retained only when both the shifted X return and the
        Y return are fully observed by that horizon. This keeps final-window
        covariance aligned with the available variance components.

    Performance convention:
      * x_variance may carry the X variance/count precomputed once for this
        X/window/lag and shared across every Y. Omitting it preserves the public
        helper's standalone behavior.
    """
    if x_variance is None:
        var_x, n_x = variance_by_shifted_end(
            x_end,
            x_ret,
            lag_ns,
            window_start_ns,
            window_end_ns,
        )
    else:
        var_x, n_x = x_variance
    if y_window is None:
        y_window = prepare_y_window(
            y_start,
            y_end,
            y_ret,
            window_start_ns,
            window_end_ns,
        )
    var_y = y_window.var_y
    n_y = y_window.n_y

    if len(x_ret) == 0 or len(y_ret) == 0:
        return 0.0, var_x, var_y, 0, n_x, n_y

    x_first, x_stop = overlap_candidate_bounds(
        x_start,
        x_end,
        lag_ns,
        window_start_ns,
        window_end_ns,
    )
    y_first = y_window.cov_first
    y_stop = y_window.cov_stop

    if x_first >= x_stop or y_first >= y_stop:
        return 0.0, var_x, var_y, 0, n_x, n_y

    cov = 0.0
    n_overlap = 0
    j = y_first

    # Both series are sorted by end time; because intervals are consecutive per
    # instrument, start times are also nondecreasing. This two-pointer sweep avoids
    # the Cartesian product. Shifting X scalars here avoids allocating two full
    # shifted timestamp arrays for every lag.
    for i in range(x_first, x_stop):
        xs_i = int(x_start[i]) + lag_ns
        xe_i = int(x_end[i]) + lag_ns
        # At the final analysis boundary, overlap_end alone is insufficient:
        # min(x_end, y_end) can be inside the horizon while the other return is
        # not fully observed until after it. Such a covariance would have no
        # matching variance contribution anywhere in this run.
        if analysis_end_ns is not None and xe_i > analysis_end_ns:
            break

        while j < y_stop and int(y_end[j]) <= xs_i:
            j += 1

        k = j
        while k < y_stop and int(y_start[k]) < xe_i:
            if analysis_end_ns is not None and int(y_end[k]) > analysis_end_ns:
                break
            overlap_start = max(xs_i, int(y_start[k]))
            overlap_end = min(xe_i, int(y_end[k]))

            if (
                overlap_start < overlap_end
                and window_start_ns < overlap_end <= window_end_ns
            ):
                cov += float(x_ret[i] * y_ret[k])
                n_overlap += 1

            k += 1

    return cov, var_x, var_y, n_overlap, n_x, n_y


def select_group(
    rows: list[dict[str, str]],
    group: str | None,
    market_type: str | None,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if group and row.get("group") != group:
            continue
        if market_type and row.get("market_type") != market_type:
            continue
        out.append(row)
    return out


def read_universe_limited(path: str, max_rows: int | None) -> list[dict[str, str]]:
    """Read a small control CSV without allowing it to consume unbounded RAM."""
    rows: list[dict[str, str]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"exchange", "market_type", "symbol", "group"}
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise RuntimeError(
                f"Universe CSV {path} is missing columns {sorted(missing)}"
            )
        for row in reader:
            rows.append(row)
            if max_rows is not None and len(rows) > max_rows:
                raise RuntimeError(
                    f"Universe CSV exceeds --max-universe-rows={max_rows:,}; "
                    "split the job or raise the guard deliberately"
                )
    return rows


def require_unique_instruments(
    rows: list[dict[str, str]],
    group_label: str,
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = instrument_key(row)
        if key in seen:
            raise RuntimeError(
                f"Duplicate instrument {key} in selected {group_label} group; "
                "each exchange/market_type/symbol may appear only once per group"
            )
        seen.add(key)


def base_asset_key(row: dict[str, str], group_label: str) -> str:
    value = (row.get("base_asset") or "").strip()
    if not value:
        raise RuntimeError(
            f"Instrument {instrument_key(row)} in selected {group_label} group "
            "has no base_asset; it is required by the default equal-base pairing"
        )
    return value.upper()


def is_same_instrument(x: dict[str, str], y: dict[str, str]) -> bool:
    return (
        x.get("exchange") == y.get("exchange")
        and x.get("market_type") == y.get("market_type")
        and x.get("symbol") == y.get("symbol")
    )


def iter_windows(start, end, window_hours: float) -> Iterable[tuple[object, object, int, int]]:
    w = start
    while w < end:
        w2 = min(w + timedelta(hours=window_hours), end)
        yield w, w2, dt_to_ns(w), dt_to_ns(w2)
        w = w2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast, 8GB-safe A x B Hayashi-Yoshida lag components from "
            "validated interval caches. Per-window rows are additive accounting "
            "components, not causal rolling estimates."
        )
    )
    parser.add_argument("--universe", required=True)
    parser.add_argument("--interval-root", required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="Output parquet or csv path. .parquet recommended.",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--window-hours",
        type=float,
        default=2.0,
        help="Window size in hours; decimals such as 0.25 are allowed.",
    )
    parser.add_argument("--max-lag-ms", type=int, default=100)
    parser.add_argument("--lag-step-ms", type=int, default=10)
    parser.add_argument(
        "--boundary-ms",
        type=int,
        default=0,
        help=(
            "Extra read buffer around each HY window. Usually 0 is enough because "
            "the manifest max_interval_ms is used for partition discovery. This "
            "does not expand the estimator or strict manifest horizon."
        ),
    )
    parser.add_argument("--price-mode", choices=["mid", "microprice"], default="mid")
    parser.add_argument("--a-group", default="A")
    parser.add_argument("--b-group", default="B")
    parser.add_argument("--a-market-type")
    parser.add_argument("--b-market-type")
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--skip-self-pairs", action="store_true")
    parser.add_argument(
        "--pair-base-asset-match",
        choices=["equal", "any"],
        default="equal",
        help=(
            "Pair only instruments whose normalized base_asset values are equal. "
            "This filtering happens before pair-offset/pair-limit; 'any' is an "
            "explicit unsafe override for full A x B."
        ),
    )
    parser.add_argument(
        "--max-universe-rows",
        type=int,
        default=10_000,
        help="Reject an unexpectedly large universe control file before it fills RAM.",
    )
    parser.add_argument(
        "--max-selected-pairs",
        type=int,
        default=10_000,
        help="Hard cap on pairs in one shard, independent of output row count.",
    )
    parser.add_argument(
        "--max-selected-instruments",
        type=int,
        default=2_000,
        help="Hard cap on unique caches validated by one process.",
    )
    parser.add_argument(
        "--max-cache-files",
        type=int,
        default=20_000,
        help=(
            "Maximum Parquet files for one instrument/range. Protects Python and "
            "DuckDB metadata RAM from pathological small-file caches."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default="2GB",
        help="DuckDB-only limit. Default 2GB is conservative for an 8GB server.",
    )
    parser.add_argument("--temp-dir", default="./duckdb_tmp")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--max-temp-size",
        default="8GB",
        help="DuckDB spill limit, preventing temp files from filling the disk.",
    )
    parser.add_argument(
        "--max-base-numpy-mib",
        type=float,
        default=512.0,
        help=(
            "Fail before fetching a window whose six timestamp/return arrays "
            "would exceed this MiB limit. Set 0 only to disable deliberately."
        ),
    )
    parser.add_argument(
        "--max-window-work-items",
        type=int,
        default=50_000_000,
        help="Fail if lag_count*(x_rows+y_rows) exceeds this CPU guard; 0 disables.",
    )
    parser.add_argument(
        "--max-total-work-items",
        type=int,
        default=2_000_000_000,
        help="Abort a shard when cumulative estimated HY work exceeds this guard.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=24.0,
        help="Abort cleanly after this wall-clock runtime; 0 disables.",
    )
    parser.add_argument(
        "--max-lag-count",
        type=int,
        default=201,
        help="Reject unexpectedly dense lag grids before allocation.",
    )
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=5_000_000,
        help="Reject oversized shards before writing; 0 disables.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=10.0,
        help="Abort when output or temp filesystem free space drops below this value.",
    )
    parser.add_argument(
        "--cache-validation",
        choices=["strict", "warn", "off"],
        default="strict",
        help=(
            "Validate manifest identity/range and chunk filename coverage before "
            "opening outputs. strict is recommended for production."
        ),
    )
    parser.add_argument(
        "--missing-cache-policy",
        choices=["error", "warn", "empty"],
        default="error",
        help="What to do when a runtime window has no cache files.",
    )
    parser.add_argument(
        "--allow-unlabeled-legacy-price-mode",
        action="store_true",
        help=(
            "Unsafe compatibility override: trust a legacy cache without a physical "
            "price_mode column as the requested mode."
        ),
    )
    parser.add_argument(
        "--log-window-rows",
        action="store_true",
        help="Print loaded X/Y row counts and base NumPy size for every window.",
    )
    parser.add_argument(
        "--global-lock-file",
        default="/tmp/compute_pair_hy_lowram_8gb.global.lock",
        help=(
            "Host-wide lock that serializes jobs on an 8GB server. Pass an empty "
            "string only when an external scheduler already enforces one job."
        ),
    )
    args = parser.parse_args()

    if args.window_hours <= 0:
        parser.error("--window-hours must be > 0")
    if args.max_lag_ms < 0:
        parser.error("--max-lag-ms must be >= 0")
    if args.lag_step_ms <= 0:
        parser.error("--lag-step-ms must be > 0")
    if args.boundary_ms < 0:
        parser.error("--boundary-ms must be >= 0")
    if args.pair_offset < 0:
        parser.error("--pair-offset must be >= 0")
    if args.pair_limit is not None and args.pair_limit <= 0:
        parser.error("--pair-limit must be > 0 when provided")
    if args.max_universe_rows <= 0:
        parser.error("--max-universe-rows must be > 0")
    if args.max_selected_pairs <= 0:
        parser.error("--max-selected-pairs must be > 0")
    if args.max_selected_instruments <= 0:
        parser.error("--max-selected-instruments must be > 0")
    if args.max_cache_files <= 0:
        parser.error("--max-cache-files must be > 0")
    if args.threads <= 0:
        parser.error("--threads must be > 0")
    if args.max_base_numpy_mib < 0:
        parser.error("--max-base-numpy-mib must be >= 0")
    if args.max_window_work_items < 0:
        parser.error("--max-window-work-items must be >= 0")
    if args.max_total_work_items < 0:
        parser.error("--max-total-work-items must be >= 0")
    if args.max_runtime_hours < 0:
        parser.error("--max-runtime-hours must be >= 0")
    if args.max_lag_count <= 0:
        parser.error("--max-lag-count must be > 0")
    if args.max_output_rows < 0:
        parser.error("--max-output-rows must be >= 0")
    if args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb must be >= 0")
    if args.max_lag_ms % args.lag_step_ms != 0:
        parser.error(
            "--max-lag-ms must be divisible by --lag-step-ms so the lag grid is "
            "symmetric and contains zero"
        )

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    if start is None or end is None or start >= end:
        parser.error("--start and --end must define a non-empty UTC range")

    rows = read_universe_limited(args.universe, args.max_universe_rows)
    group_a = select_group(rows, args.a_group, args.a_market_type)
    group_b = select_group(rows, args.b_group, args.b_market_type)
    if not group_a:
        parser.error(
            f"No A instruments selected. Check group={args.a_group!r} "
            "and --a-market-type."
        )
    if not group_b:
        parser.error(
            f"No B instruments selected. Check group={args.b_group!r} "
            "and --b-market-type."
        )
    try:
        require_unique_instruments(group_a, "A")
        require_unique_instruments(group_b, "B")
    except RuntimeError as exc:
        parser.error(str(exc))

    raw_cross_pair_count = len(group_a) * len(group_b)
    eligible_pair_groups: list[
        tuple[dict[str, str], list[dict[str, str]]]
    ] = []
    if args.pair_base_asset_match == "equal":
        try:
            b_by_base: dict[str, list[dict[str, str]]] = {}
            for y in group_b:
                b_by_base.setdefault(base_asset_key(y, "B"), []).append(y)
            for x in group_a:
                matching_y = b_by_base.get(base_asset_key(x, "A"), [])
                if matching_y:
                    eligible_pair_groups.append((x, matching_y))
        except RuntimeError as exc:
            parser.error(str(exc))
    else:
        eligible_pair_groups = [(x, group_b) for x in group_a]

    eligible_pair_count = sum(
        len(matching_y) for _, matching_y in eligible_pair_groups
    )
    selection_stop = (
        eligible_pair_count
        if args.pair_limit is None
        else min(eligible_pair_count, args.pair_offset + args.pair_limit)
    )

    def make_pairs():
        # Offset and limit apply to the already-filtered, X-major pair stream.
        skip = args.pair_offset
        remaining = max(0, selection_stop - args.pair_offset)
        for x, matching_y in eligible_pair_groups:
            if remaining <= 0:
                return
            if skip >= len(matching_y):
                skip -= len(matching_y)
                continue
            first = skip
            take = min(remaining, len(matching_y) - first)
            for y in matching_y[first : first + take]:
                yield x, y
            remaining -= take
            skip = 0

    selected_pair_count = max(0, selection_stop - args.pair_offset)
    if selected_pair_count <= 0:
        parser.error(
            "Pair selection is empty after base_asset filtering; check "
            "base_asset values and --pair-offset/--pair-limit"
        )
    if selected_pair_count > args.max_selected_pairs:
        parser.error(
            f"Selected {selected_pair_count:,} pairs, exceeding "
            f"--max-selected-pairs={args.max_selected_pairs:,}. Use --pair-limit "
            "to shard the job."
        )
    print(
        f"[PREFLIGHT] raw_AxB_pairs={raw_cross_pair_count:,} "
        f"base_asset_match={args.pair_base_asset_match} "
        f"eligible_pairs={eligible_pair_count:,} "
        f"selected_after_offset_limit={selected_pair_count:,}"
    )

    lag_values = list(
        range(-args.max_lag_ms, args.max_lag_ms + 1, args.lag_step_ms)
    )
    if len(lag_values) > args.max_lag_count:
        parser.error(
            f"Lag grid has {len(lag_values)} values, exceeding --max-lag-count="
            f"{args.max_lag_count}"
        )
    max_lag_ns = args.max_lag_ms * 1_000_000
    boundary_ns = args.boundary_ms * 1_000_000
    analysis_start_ns = dt_to_ns(start)
    analysis_end_ns = dt_to_ns(end)

    duration_seconds = (end - start).total_seconds()
    window_count = math.ceil(duration_seconds / (args.window_hours * 3600.0))
    raw_output_upper_bound = selected_pair_count * window_count * len(lag_values)
    if args.max_output_rows > 0 and raw_output_upper_bound > args.max_output_rows:
        parser.error(
            f"Selected pairs imply up to {raw_output_upper_bound:,} component rows, "
            f"exceeding --max-output-rows={args.max_output_rows:,}. Reduce "
            "--pair-limit before the pair preflight."
        )

    selected_instruments: dict[tuple[str, str, str], dict[str, str]] = {}
    selected_x_keys: set[tuple[str, str, str]] = set()
    selected_y_keys: set[tuple[str, str, str]] = set()
    runnable_pairs: list[
        tuple[int, dict[str, str], dict[str, str]]
    ] = []
    for pair_i, (x, y) in enumerate(make_pairs(), 1):
        if args.skip_self_pairs and is_same_instrument(x, y):
            continue
        runnable_pairs.append((pair_i, x, y))
        x_key = instrument_key(x)
        y_key = instrument_key(y)
        selected_x_keys.add(x_key)
        selected_y_keys.add(y_key)
        selected_instruments[x_key] = x
        selected_instruments[y_key] = y
        if len(selected_instruments) > args.max_selected_instruments:
            parser.error(
                f"Selected more than {args.max_selected_instruments:,} unique "
                "instruments; split the pair shard"
            )
    runnable_pair_count = len(runnable_pairs)
    if runnable_pair_count == 0:
        parser.error("All selected pairs were removed by --skip-self-pairs")

    # The selected product order is X-major. Group contiguous pairs by X so each
    # X/window is scanned, sorted, and materialized once instead of once per Y.
    x_pair_groups: list[
        tuple[dict[str, str], list[tuple[int, dict[str, str]]]]
    ] = []
    for pair_i, x, y in runnable_pairs:
        if (
            not x_pair_groups
            or instrument_key(x_pair_groups[-1][0]) != instrument_key(x)
        ):
            x_pair_groups.append((x, []))
        x_pair_groups[-1][1].append((pair_i, y))

    estimated_output_rows = runnable_pair_count * window_count * len(lag_values)
    if args.max_output_rows > 0 and estimated_output_rows > args.max_output_rows:
        parser.error(
            f"Estimated component output has {estimated_output_rows:,} rows, "
            f"exceeding --max-output-rows={args.max_output_rows:,}. Reduce "
            "--pair-limit or raise the guard only after checking disk capacity."
        )
    print(
        f"[PREFLIGHT] pairs={runnable_pair_count:,} windows={window_count:,} "
        f"lags={len(lag_values):,} estimated_component_rows="
        f"{estimated_output_rows:,} (~{estimated_output_rows * ESTIMATED_CSV_BYTES_PER_ROW / 1024 ** 3:.1f}GiB CSV)"
    )

    out_path = Path(args.out)
    if out_path.suffix.lower() not in {".csv", ".parquet"}:
        parser.error("--out must end in .csv or .parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_path.with_name(out_path.name + ".partial.csv")
    partial_parquet = out_path.with_name(out_path.name + ".partial")
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    partial_summary = summary_path.with_name(summary_path.name + ".partial")
    success_path = out_path.with_name(out_path.name + ".SUCCESS.json")
    partial_success = success_path.with_name(success_path.name + ".partial")
    lock_path = out_path.with_name(out_path.name + ".lock")

    temp_path = Path(args.temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    release_global_lock = None
    if args.global_lock_file:
        global_lock_path = Path(args.global_lock_file).expanduser()
        global_lock_path.parent.mkdir(parents=True, exist_ok=True)
        release_global_lock = acquire_lock(global_lock_path, "Global HY job")
    ensure_free_disk(out_path.parent, args.min_free_disk_gb, "output")
    ensure_free_disk(temp_path, args.min_free_disk_gb, "temp")
    ensure_startup_disk_budget(
        out_path.parent,
        temp_path,
        args.min_free_disk_gb,
        args.max_temp_size,
        estimated_output_rows,
    )
    release_output_lock = acquire_lock(lock_path, "Output")

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC';")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)};")
    con.execute(f"SET temp_directory = {sql_quote(str(temp_path))};")
    con.execute(
        f"SET max_temp_directory_size = {sql_quote(args.max_temp_size)};"
    )
    con.execute(f"SET threads = {args.threads};")
    con.execute("SET preserve_insertion_order = false;")

    if args.cache_validation != "off":
        print(
            f"[PREFLIGHT] validating {len(selected_instruments)} unique instrument "
            f"caches before creating outputs"
        )
        for key, inst in selected_instruments.items():
            used_as_x = key in selected_x_keys
            required_start_ns = (
                analysis_start_ns - max_lag_ns
                if used_as_x
                else analysis_start_ns
            )
            required_end_ns = (
                analysis_end_ns + max_lag_ns
                if used_as_x
                else analysis_end_ns
            )
            validate_cache_location(
                con,
                args.interval_root,
                inst,
                args.price_mode,
                required_start_ns,
                required_end_ns,
                policy=args.cache_validation,
                allow_unlabeled_legacy=args.allow_unlabeled_legacy_price_mode,
                max_cache_files=args.max_cache_files,
            )
        print("[PREFLIGHT] cache validation passed")

    cache_locations = {
        key: resolve_cache_location(args.interval_root, inst, args.price_mode)
        for key, inst in selected_instruments.items()
    }
    cache_identities = {
        key: cache_identity(location)
        for key, location in cache_locations.items()
    }
    excluded_ranges_by_key = {
        key: cache_excluded_ranges(location)
        for key, location in cache_locations.items()
    }
    excluded_hour_count = sum(
        len(ranges) for ranges in excluded_ranges_by_key.values()
    )
    if excluded_hour_count:
        print(
            f"[PREFLIGHT] intentional abnormal-hour exclusions="
            f"{excluded_hour_count} across selected instrument caches"
        )
    research_config = {
        "script_version": SCRIPT_VERSION,
        "price_mode": args.price_mode,
        "window_hours": args.window_hours,
        "max_lag_ms": args.max_lag_ms,
        "lag_step_ms": args.lag_step_ms,
        "boundary_ms": args.boundary_ms,
        "a_group": args.a_group,
        "b_group": args.b_group,
        "a_market_type": args.a_market_type,
        "b_market_type": args.b_market_type,
        "skip_self_pairs": args.skip_self_pairs,
        "pair_base_asset_match": args.pair_base_asset_match,
        "covariance_window_assignment": "min_shifted_x_end_y_end",
        "variance_window_assignment": "individual_end",
        "require_both_return_ends_within_analysis": True,
        "cache_coverage_rule": {
            "x_end_range": "analysis_start-max_lag through analysis_end+max_lag",
            "y_end_range": "analysis_start through analysis_end",
            "boundary_and_max_interval_do_not_extend_manifest_horizon": True,
            "intentional_excluded_hours": (
                "skip every pair/window whose role-aware interval-end range "
                "intersects an excluded source hour"
            ),
        },
        "component_corr_interpretation": "diagnostic_non_standalone",
    }
    config_hash = stable_config_hash(research_config)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + config_hash[:12]
    )

    fieldnames = [
        "run_id", "config_hash",
        "x_exchange", "x_market_type", "x_symbol", "x_base_asset",
        "y_exchange", "y_market_type", "y_symbol", "y_base_asset",
        "x_cache_layout", "y_cache_layout",
        "x_max_interval_ms", "y_max_interval_ms",
        "x_cache_config_version", "y_cache_config_version",
        "x_drop_zero_returns", "y_drop_zero_returns",
        "x_cache_config_hash", "y_cache_config_hash",
        "price_mode", "window_start", "window_end", "lag_ms",
        "cov", "var_x", "var_y", "corr", "corr_is_diagnostic",
        "n_overlap", "n_x", "n_y",
    ]
    summary_fieldnames = [
        "run_id", "config_hash",
        "x_exchange", "x_market_type", "x_symbol", "x_base_asset",
        "y_exchange", "y_market_type", "y_symbol", "y_base_asset",
        "x_cache_layout", "y_cache_layout",
        "x_max_interval_ms", "y_max_interval_ms",
        "x_cache_config_version", "y_cache_config_version",
        "x_drop_zero_returns", "y_drop_zero_returns",
        "x_cache_config_hash", "y_cache_config_hash",
        "price_mode", "lag_ms",
        "sum_cov", "sum_var_x", "sum_var_y", "agg_corr",
        "total_overlap", "total_n_x", "total_n_y", "n_windows",
    ]

    processed_pairs = runnable_pair_count
    skipped_pairs = selected_pair_count - runnable_pair_count
    max_base_bytes = (
        None
        if args.max_base_numpy_mib == 0
        else int(args.max_base_numpy_mib * 1024 ** 2)
    )
    max_work_rows = (
        None
        if args.max_window_work_items == 0
        else args.max_window_work_items // len(lag_values)
    )
    job_started_monotonic = time.monotonic()
    total_work_items = 0
    component_rows_written = 0
    skipped_pair_windows = 0

    def make_accumulator() -> PairAccumulator:
        n_lags = len(lag_values)
        return PairAccumulator(
            np.zeros(n_lags, dtype=np.float64),
            np.zeros(n_lags, dtype=np.float64),
            np.zeros(n_lags, dtype=np.float64),
            np.zeros(n_lags, dtype=np.int64),
            np.zeros(n_lags, dtype=np.int64),
            np.zeros(n_lags, dtype=np.int64),
            np.zeros(n_lags, dtype=np.int64),
        )

    def pair_identity_fields(
        x: dict[str, str],
        y: dict[str, str],
    ) -> dict[str, object]:
        x_identity = cache_identities[instrument_key(x)]
        y_identity = cache_identities[instrument_key(y)]
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "x_cache_layout": x_identity["layout"],
            "y_cache_layout": y_identity["layout"],
            "x_max_interval_ms": x_identity["max_interval_ms"],
            "y_max_interval_ms": y_identity["max_interval_ms"],
            "x_cache_config_version": x_identity["cache_config_version"],
            "y_cache_config_version": y_identity["cache_config_version"],
            "x_drop_zero_returns": x_identity["drop_zero_returns"],
            "y_drop_zero_returns": y_identity["drop_zero_returns"],
            "x_cache_config_hash": x_identity["cache_config_hash"],
            "y_cache_config_hash": y_identity["cache_config_hash"],
        }

    with (
        open(tmp_csv, "w", newline="") as f,
        open(partial_summary, "w", newline="") as summary_f,
    ):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        summary_writer = csv.DictWriter(summary_f, fieldnames=summary_fieldnames)
        summary_writer.writeheader()

        for x_group_i, (x, pair_members) in enumerate(x_pair_groups, 1):
            print(
                f"[INFO] X group {x_group_i}/{len(x_pair_groups)} "
                f"{instrument_key(x)} with {len(pair_members)} Y instruments"
            )
            accumulators = {
                pair_i: make_accumulator() for pair_i, _ in pair_members
            }
            identity_fields = {
                pair_i: pair_identity_fields(x, y) for pair_i, y in pair_members
            }
            for pair_i, y in pair_members:
                print(
                    f"[INFO] pair {pair_i}/{selected_pair_count} "
                    f"{instrument_key(x)} x {instrument_key(y)} "
                    f"price_mode={args.price_mode}"
                )

            for w, w2, w_start_ns, w_end_ns in iter_windows(
                start,
                end,
                args.window_hours,
            ):
                if (
                    args.max_runtime_hours > 0
                    and time.monotonic() - job_started_monotonic
                    > args.max_runtime_hours * 3600.0
                ):
                    raise RuntimeError(
                        f"Runtime exceeded --max-runtime-hours="
                        f"{args.max_runtime_hours:g}; split the pair/date shard"
                    )
                # Load only the data that can be needed by this window under any lag.
                # For lagged X, any lag in [-max_lag, +max_lag] can move an X interval
                # into the window, so expand both sides by max_lag. boundary_ns is only
                # an optional extra buffer.
                x_read_start_ns = w_start_ns - max_lag_ns - boundary_ns
                x_read_end_ns = w_end_ns + max_lag_ns + boundary_ns
                # Y is never shifted, so max_lag does not belong in its query
                # range. Keeping the ranges role-specific reduces Parquet scans.
                y_read_start_ns = w_start_ns - boundary_ns
                y_read_end_ns = w_end_ns + boundary_ns

                x_excluded_ranges = excluded_ranges_by_key.get(
                    instrument_key(x), ()
                )
                if ranges_overlap(
                    x_excluded_ranges,
                    w_start_ns - max_lag_ns,
                    w_end_ns + max_lag_ns,
                ):
                    skipped_pair_windows += len(pair_members)
                    print(
                        f"[SKIP] X abnormal-hour coverage intersects window "
                        f"{w.isoformat(sep=' ')} -> {w2.isoformat(sep=' ')} "
                        f"for {instrument_key(x)}; skipped "
                        f"{len(pair_members)} pair-window(s)"
                    )
                    continue

                x_limits = []
                if max_base_bytes is not None:
                    x_limits.append(max_base_bytes // 24)
                if max_work_rows is not None:
                    x_limits.append(max_work_rows)
                x_row_limit = min(x_limits) if x_limits else None

                x_start, x_end, x_ret = load_intervals(
                    con,
                    args.interval_root,
                    x,
                    x_read_start_ns,
                    x_read_end_ns,
                    args.price_mode,
                    missing_cache_policy=args.missing_cache_policy,
                    allow_unlabeled_legacy=args.allow_unlabeled_legacy_price_mode,
                    max_rows=x_row_limit,
                    max_cache_files=args.max_cache_files,
                    max_end_ns=analysis_end_ns + max_lag_ns,
                )
                used_base_bytes = arrays_nbytes(x_start, x_end, x_ret)
                y_limits = []
                if max_base_bytes is not None:
                    y_limits.append(max(0, (max_base_bytes - used_base_bytes) // 24))
                if max_work_rows is not None:
                    y_limits.append(max(0, max_work_rows - len(x_ret)))
                y_row_limit = min(y_limits) if y_limits else None
                x_variance_by_lag = [
                    variance_by_shifted_end(
                        x_end,
                        x_ret,
                        lag_ms * 1_000_000,
                        w_start_ns,
                        w_end_ns,
                    )
                    for lag_ms in lag_values
                ]
                for pair_i, y in pair_members:
                    if (
                        args.max_runtime_hours > 0
                        and time.monotonic() - job_started_monotonic
                        > args.max_runtime_hours * 3600.0
                    ):
                        raise RuntimeError(
                            f"Runtime exceeded --max-runtime-hours="
                            f"{args.max_runtime_hours:g}; split the pair/date shard"
                        )

                    if ranges_overlap(
                        excluded_ranges_by_key.get(instrument_key(y), ()),
                        w_start_ns,
                        w_end_ns,
                    ):
                        skipped_pair_windows += 1
                        print(
                            f"[SKIP] Y abnormal-hour coverage intersects window "
                            f"{w.isoformat(sep=' ')} -> {w2.isoformat(sep=' ')} "
                            f"for pair {instrument_key(x)} x {instrument_key(y)}"
                        )
                        continue

                    y_start, y_end, y_ret = load_intervals(
                        con,
                        args.interval_root,
                        y,
                        y_read_start_ns,
                        y_read_end_ns,
                        args.price_mode,
                        missing_cache_policy=args.missing_cache_policy,
                        allow_unlabeled_legacy=args.allow_unlabeled_legacy_price_mode,
                        max_rows=y_row_limit,
                        max_cache_files=args.max_cache_files,
                        max_end_ns=analysis_end_ns,
                    )

                    base_numpy_bytes = arrays_nbytes(
                        x_start, x_end, x_ret, y_start, y_end, y_ret
                    )
                    work_items = len(lag_values) * (len(x_ret) + len(y_ret))
                    if (
                        args.max_window_work_items > 0
                        and work_items > args.max_window_work_items
                    ):
                        raise RuntimeError(
                            f"Estimated HY work {work_items:,} exceeds "
                            f"--max-window-work-items="
                            f"{args.max_window_work_items:,} for "
                            f"{instrument_key(x)} x {instrument_key(y)} in "
                            f"{w} -> {w2}. Reduce --window-hours or lag count."
                        )
                    total_work_items += work_items
                    if (
                        args.max_total_work_items > 0
                        and total_work_items > args.max_total_work_items
                    ):
                        raise RuntimeError(
                            f"Cumulative estimated HY work reached "
                            f"{total_work_items:,}, exceeding "
                            f"--max-total-work-items="
                            f"{args.max_total_work_items:,}. Split by pair or date."
                        )

                    if args.log_window_rows:
                        base_numpy_mib = base_numpy_bytes / (1024 ** 2)
                        print(
                            f"[WINDOW] {w.isoformat(sep=' ')} -> "
                            f"{w2.isoformat(sep=' ')} "
                            f"pair={instrument_key(x)}x{instrument_key(y)} "
                            f"x_rows={len(x_ret):,} y_rows={len(y_ret):,} "
                            f"base_numpy={base_numpy_mib:.1f}MiB"
                        )

                    y_window = prepare_y_window(
                        y_start,
                        y_end,
                        y_ret,
                        w_start_ns,
                        w_end_ns,
                    )
                    accumulator = accumulators[pair_i]
                    row_identity = identity_fields[pair_i]

                    for lag_i, lag_ms in enumerate(lag_values):
                        (
                            cov,
                            var_x,
                            var_y,
                            n_overlap,
                            n_x,
                            n_y,
                        ) = hy_for_lag_boundary_fixed(
                            x_start,
                            x_end,
                            x_ret,
                            y_start,
                            y_end,
                            y_ret,
                            lag_ms * 1_000_000,
                            w_start_ns,
                            w_end_ns,
                            y_window=y_window,
                            analysis_end_ns=analysis_end_ns,
                            x_variance=x_variance_by_lag[lag_i],
                        )
                        corr = corr_or_none(cov, var_x, var_y)

                        accumulator.sum_cov[lag_i] += cov
                        accumulator.sum_var_x[lag_i] += var_x
                        accumulator.sum_var_y[lag_i] += var_y
                        accumulator.total_overlap[lag_i] += n_overlap
                        accumulator.total_n_x[lag_i] += n_x
                        accumulator.total_n_y[lag_i] += n_y
                        accumulator.n_windows[lag_i] += 1

                        writer.writerow({
                            **row_identity,
                            "x_exchange": x["exchange"],
                            "x_market_type": x["market_type"],
                            "x_symbol": x["symbol"],
                            "x_base_asset": (x.get("base_asset") or "").strip().upper(),
                            "y_exchange": y["exchange"],
                            "y_market_type": y["market_type"],
                            "y_symbol": y["symbol"],
                            "y_base_asset": (y.get("base_asset") or "").strip().upper(),
                            "price_mode": args.price_mode,
                            "window_start": w.isoformat(sep=" "),
                            "window_end": w2.isoformat(sep=" "),
                            "lag_ms": lag_ms,
                            "cov": cov,
                            "var_x": var_x,
                            "var_y": var_y,
                            "corr": corr,
                            "corr_is_diagnostic": True,
                            "n_overlap": n_overlap,
                            "n_x": n_x,
                            "n_y": n_y,
                        })
                        component_rows_written += 1

                    del y_start, y_end, y_ret
                    del y_window

                f.flush()
                ensure_free_disk(out_path.parent, args.min_free_disk_gb, "output")
                ensure_free_disk(temp_path, args.min_free_disk_gb, "temp")

                # Keep X only while every Y for this X/window is processed.
                del x_variance_by_lag
                del x_start, x_end, x_ret

            for pair_i, y in pair_members:
                accumulator = accumulators[pair_i]
                row_identity = identity_fields[pair_i]
                for lag_i, lag_ms in enumerate(lag_values):
                    sx = float(accumulator.sum_var_x[lag_i])
                    sy = float(accumulator.sum_var_y[lag_i])
                    sc = float(accumulator.sum_cov[lag_i])
                    agg_corr = corr_or_none(sc, sx, sy)

                    summary_writer.writerow({
                        **row_identity,
                        "x_exchange": x["exchange"],
                        "x_market_type": x["market_type"],
                        "x_symbol": x["symbol"],
                        "x_base_asset": (x.get("base_asset") or "").strip().upper(),
                        "y_exchange": y["exchange"],
                        "y_market_type": y["market_type"],
                        "y_symbol": y["symbol"],
                        "y_base_asset": (y.get("base_asset") or "").strip().upper(),
                        "price_mode": args.price_mode,
                        "lag_ms": lag_ms,
                        "sum_cov": sc,
                        "sum_var_x": sx,
                        "sum_var_y": sy,
                        "agg_corr": agg_corr,
                        "total_overlap": int(
                            accumulator.total_overlap[lag_i]
                        ),
                        "total_n_x": int(accumulator.total_n_x[lag_i]),
                        "total_n_y": int(accumulator.total_n_y[lag_i]),
                        "n_windows": int(accumulator.n_windows[lag_i]),
                    })

            summary_f.flush()
            ensure_free_disk(out_path.parent, args.min_free_disk_gb, "output")
            ensure_free_disk(temp_path, args.min_free_disk_gb, "temp")
            del accumulators, identity_fields

    component_columns = """{
        'run_id': 'VARCHAR', 'config_hash': 'VARCHAR',
        'x_exchange': 'VARCHAR', 'x_market_type': 'VARCHAR', 'x_symbol': 'VARCHAR',
        'x_base_asset': 'VARCHAR',
        'y_exchange': 'VARCHAR', 'y_market_type': 'VARCHAR', 'y_symbol': 'VARCHAR',
        'y_base_asset': 'VARCHAR',
        'x_cache_layout': 'VARCHAR', 'y_cache_layout': 'VARCHAR',
        'x_max_interval_ms': 'DOUBLE', 'y_max_interval_ms': 'DOUBLE',
        'x_cache_config_version': 'BIGINT', 'y_cache_config_version': 'BIGINT',
        'x_drop_zero_returns': 'BOOLEAN', 'y_drop_zero_returns': 'BOOLEAN',
        'x_cache_config_hash': 'VARCHAR', 'y_cache_config_hash': 'VARCHAR',
        'price_mode': 'VARCHAR', 'window_start': 'TIMESTAMP',
        'window_end': 'TIMESTAMP', 'lag_ms': 'BIGINT', 'cov': 'DOUBLE',
        'var_x': 'DOUBLE', 'var_y': 'DOUBLE', 'corr': 'DOUBLE',
        'corr_is_diagnostic': 'BOOLEAN',
        'n_overlap': 'BIGINT', 'n_x': 'BIGINT', 'n_y': 'BIGINT'
    }"""
    components_scan = (
        f"read_csv({sql_quote(str(tmp_csv))}, header=true, "
        f"columns={component_columns}, nullstr='')"
    )

    # Remove the old completion marker only when the new artifacts are ready to
    # publish. If replacement is interrupted, absence of this marker exposes the
    # incomplete generation instead of silently presenting it as complete.
    if success_path.exists():
        success_path.unlink()

    if out_path.suffix.lower() == ".parquet":
        ensure_parquet_conversion_space(
            out_path.parent,
            tmp_csv,
            args.min_free_disk_gb,
        )
        if partial_parquet.exists():
            partial_parquet.unlink()
        con.execute(f"""
        COPY (
            SELECT *
            FROM {components_scan}
        )
        TO {sql_quote(str(partial_parquet))}
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        partial_parquet.replace(out_path)
        tmp_csv.unlink()
    else:
        tmp_csv.replace(out_path)

    partial_summary.replace(summary_path)

    partial_success.write_text(
        json.dumps(
            {
                "status": "complete",
                "script_version": SCRIPT_VERSION,
                "duckdb_version": duckdb.__version__,
                "run_id": run_id,
                "config_hash": config_hash,
                "completed_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "components": str(out_path),
                "summary": str(summary_path),
                "start": start.isoformat(timespec="microseconds"),
                "end": end.isoformat(timespec="microseconds"),
                "price_mode": args.price_mode,
                "raw_AxB_pairs": raw_cross_pair_count,
                "eligible_base_asset_pairs": eligible_pair_count,
                "pair_base_asset_match": args.pair_base_asset_match,
                "processed_pairs": processed_pairs,
                "component_rows": component_rows_written,
                "estimated_component_rows_before_exclusions": estimated_output_rows,
                "lag_count": len(lag_values),
                "window_count": window_count,
                "skipped_pair_windows": skipped_pair_windows,
                "estimated_total_work_items": total_work_items,
                "research_config": research_config,
                "full_arguments": vars(args),
                "cache_identities": [
                    {
                        "exchange": key[0],
                        "market_type": key[1],
                        "symbol": key[2],
                        **identity,
                    }
                    for key, identity in sorted(cache_identities.items())
                ],
                "component_primary_key": [
                    "x_exchange",
                    "x_market_type",
                    "x_symbol",
                    "x_base_asset",
                    "y_exchange",
                    "y_market_type",
                    "y_symbol",
                    "y_base_asset",
                    "price_mode",
                    "window_start",
                    "window_end",
                    "lag_ms",
                ],
                "merge_requirements": [
                    "all config_hash values must match",
                    "cache_config_hash must match for each instrument",
                    "component_primary_key must be globally unique before summing",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    partial_success.replace(success_path)
    release_output_lock()
    atexit.unregister(release_output_lock)
    if release_global_lock is not None:
        release_global_lock()
        atexit.unregister(release_global_lock)

    print(
        f"[DONE] selected_pairs={selected_pair_count} "
        f"runnable_pairs={runnable_pair_count} processed_pairs={processed_pairs} "
        f"skipped_pairs={skipped_pairs} "
        f"skipped_pair_windows={skipped_pair_windows}"
    )
    print(f"[DONE] wrote components: {out_path}")
    print(f"[DONE] wrote summary: {summary_path}")
    print(
        f"[DONE] processed requested UTC range {start} -> {end}; "
        f"intentional abnormal-hour pair-windows were excluded; per-window "
        f"loading uses ±{args.max_lag_ms + args.boundary_ms}ms"
    )


if __name__ == "__main__":
    main()
