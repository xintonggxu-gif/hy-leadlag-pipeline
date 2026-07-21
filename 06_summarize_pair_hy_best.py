#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import glob
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from common import sql_quote


SCRIPT_VERSION = "3.3.0"

PAIR_ID_COLS = [
    "x_exchange", "x_market_type", "x_symbol",
    "y_exchange", "y_market_type", "y_symbol",
]

PAIR_COLS = [
    "x_exchange", "x_market_type", "x_symbol", "x_base_asset",
    "y_exchange", "y_market_type", "y_symbol", "y_base_asset",
]

GROUP_COLS = PAIR_COLS + ["price_mode", "config_hash"]

CACHE_META_COLS = [
    "x_cache_layout", "y_cache_layout",
    "x_max_interval_ms", "y_max_interval_ms",
    "x_cache_config_version", "y_cache_config_version",
    "x_drop_zero_returns", "y_drop_zero_returns",
    "x_cache_config_hash", "y_cache_config_hash",
]

PRIMARY_KEY_COLS = PAIR_ID_COLS + [
    "price_mode", "window_start", "window_end", "lag_ms",
]

REQUIRED_COMPONENT_COLS = set(
    PRIMARY_KEY_COLS
    + CACHE_META_COLS
    + [
        "run_id", "config_hash", "x_base_asset", "y_base_asset",
        "cov", "var_x", "var_y",
        "n_overlap", "n_x", "n_y", "corr_is_diagnostic",
    ]
)


def comma(cols: list[str], indent: str = "        ") -> str:
    return (",\n" + indent).join(cols)


def prefixed(cols: list[str], alias: str) -> list[str]:
    return [f"{alias}.{c}" for c in cols]


def using_clause(cols: list[str]) -> str:
    return "(\n        " + ", ".join(cols) + "\n    )"


def sql_list(paths: list[str]) -> str:
    return "[" + ", ".join(sql_quote(path) for path in paths) + "]"


def version_tuple(value: object) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(value or ""))
    if match is None:
        return 0, 0, 0
    return tuple(int(part) for part in match.groups())


def discover_components(pattern: str, max_files: int) -> list[Path]:
    paths = sorted(
        {
            Path(match).resolve()
            for match in glob.glob(pattern, recursive=True)
            if Path(match).is_file() and Path(match).suffix.lower() == ".parquet"
        }
    )
    if not paths:
        raise RuntimeError(f"No component Parquet files matched: {pattern}")
    if len(paths) > max_files:
        raise RuntimeError(
            f"Matched {len(paths):,} Parquet files, exceeding "
            f"--max-component-files={max_files:,}"
        )
    return paths


def count_pairs_csv(path: str | Path) -> int:
    pairs_path = Path(path).expanduser().resolve()
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)
    with pairs_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "x_exchange",
            "x_market_type",
            "x_symbol",
            "y_exchange",
            "y_market_type",
            "y_symbol",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"pairs CSV {pairs_path} is missing columns {sorted(missing)}"
            )
        pairs = {
            tuple(row[column] for column in PAIR_ID_COLS)
            for row in reader
        }
    if not pairs:
        raise RuntimeError(f"pairs CSV contains no pairs: {pairs_path}")
    return len(pairs)


def load_success_markers(
    component_paths: list[Path],
    allow_cross_base_asset: bool,
) -> dict[str, dict]:
    markers: dict[str, dict] = {}
    for path in component_paths:
        marker_path = path.with_name(path.name + ".SUCCESS.json")
        if not marker_path.is_file():
            raise RuntimeError(
                f"Missing completion marker for {path}: expected {marker_path}. "
                "Only complete compute outputs may be summarized."
            )
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read {marker_path}: {exc}") from exc
        if not isinstance(marker, dict) or marker.get("status") != "complete":
            raise RuntimeError(f"Invalid or incomplete marker: {marker_path}")
        if version_tuple(marker.get("script_version")) < (3, 2, 0):
            raise RuntimeError(
                f"{marker_path} was produced by compute version "
                f"{marker.get('script_version')!r}; version 3.2.0+ is required"
            )
        for field in ("run_id", "config_hash", "component_rows"):
            if marker.get(field) in (None, ""):
                raise RuntimeError(f"Missing {field!r} in {marker_path}")
        recorded_name = Path(str(marker.get("components") or "")).name
        if recorded_name != path.name:
            raise RuntimeError(
                f"Marker {marker_path} names component {recorded_name!r}, "
                f"not {path.name!r}"
            )
        research_config = marker.get("research_config") or {}
        if (
            not allow_cross_base_asset
            and research_config.get("pair_base_asset_match") != "equal"
        ):
            raise RuntimeError(
                f"{marker_path} was not computed with equal base_asset pairing"
            )
        markers[str(path)] = marker
    return markers


def acquire_lock(path: Path, label: str):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{label} lock already exists: {path}. Verify that no task is "
            "running before removing a stale lock."
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


def ensure_free_disk(path: Path, min_free_gb: float, label: str) -> None:
    if min_free_gb <= 0:
        return
    free = shutil.disk_usage(path).free
    required = int(min_free_gb * 1024 ** 3)
    if free < required:
        raise RuntimeError(
            f"Only {free / 1024 ** 3:.1f}GiB free on {label} filesystem "
            f"at {path}; require {min_free_gb:.1f}GiB"
        )


def write_table_partial(
    con: duckdb.DuckDBPyConnection,
    table: str,
    final_path: Path,
) -> Path:
    partial = final_path.with_name(final_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    if final_path.suffix.lower() == ".parquet":
        con.execute(
            f"COPY {table} TO {sql_quote(str(partial))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD);"
        )
    else:
        con.execute(
            f"COPY {table} TO {sql_quote(str(partial))} "
            "(HEADER, DELIMITER ',');"
        )
    return partial


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and summarize compute v3.2+ HY components into "
            "lag, best-absolute-correlation, and best-positive-correlation tables."
        )
    )
    parser.add_argument(
        "--components",
        required=True,
        help="Quoted Parquet path or glob from 05_compute_pair_hy_lowram_safe_8gb.py",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--min-overlap", type=int, default=100)
    parser.add_argument("--min-n-x", type=int, default=0)
    parser.add_argument("--min-n-y", type=int, default=0)
    parser.add_argument("--min-window-rows", type=int, default=1)
    parser.add_argument("--min-abs-lag-ms", type=int, default=0)
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=0,
        help=(
            "Require exactly this many distinct pairs. 0 uses --pairs-csv when "
            "provided, otherwise disables the count check."
        ),
    )
    parser.add_argument(
        "--pairs-csv",
        help="Optional pairs.csv from stage 01; derives the expected pair count.",
    )
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--max-component-files", type=int, default=1_000)
    parser.add_argument("--max-component-rows", type=int, default=50_000_000)
    parser.add_argument(
        "--allow-cross-base-asset",
        action="store_true",
        help="Unsafe override for research that intentionally mixes base assets.",
    )
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--temp-dir", default="./duckdb_tmp")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-temp-size", default="8GB")
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument(
        "--global-lock-file",
        default="/tmp/compute_pair_hy_lowram_8gb.global.lock",
        help="Shared lock that prevents compute and summarize running together.",
    )
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be > 0")
    if args.min_overlap < 0:
        parser.error("--min-overlap must be >= 0")
    if args.min_n_x < 0 or args.min_n_y < 0:
        parser.error("--min-n-x and --min-n-y must be >= 0")
    if args.min_window_rows <= 0:
        parser.error("--min-window-rows must be > 0")
    if args.min_abs_lag_ms < 0:
        parser.error("--min-abs-lag-ms must be >= 0")
    if args.expected_pairs < 0:
        parser.error("--expected-pairs must be >= 0")
    if args.max_component_files <= 0:
        parser.error("--max-component-files must be > 0")
    if args.max_component_rows <= 0:
        parser.error("--max-component-rows must be > 0")
    if args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb must be >= 0")

    pairs_csv_count = count_pairs_csv(args.pairs_csv) if args.pairs_csv else 0
    if (
        args.expected_pairs > 0
        and pairs_csv_count > 0
        and args.expected_pairs != pairs_csv_count
    ):
        parser.error(
            f"--expected-pairs={args.expected_pairs} disagrees with "
            f"--pairs-csv count={pairs_csv_count}"
        )
    effective_expected_pairs = args.expected_pairs or pairs_csv_count

    component_paths = discover_components(args.components, args.max_component_files)
    markers = load_success_markers(component_paths, args.allow_cross_base_asset)
    marker_config_hashes = {str(marker["config_hash"]) for marker in markers.values()}
    if len(marker_config_hashes) != 1:
        raise RuntimeError(
            f"Input SUCCESS markers contain different config_hash values: "
            f"{sorted(marker_config_hashes)}"
        )
    config_hash = next(iter(marker_config_hashes))
    if args.expected_config_hash and config_hash != args.expected_config_hash:
        raise RuntimeError(
            f"config_hash={config_hash} does not match --expected-config-hash="
            f"{args.expected_config_hash}"
        )
    marker_lag_counts = {int(marker.get("lag_count", -1)) for marker in markers.values()}
    if len(marker_lag_counts) != 1 or next(iter(marker_lag_counts)) <= 0:
        raise RuntimeError(
            f"Input SUCCESS markers disagree on lag_count: {sorted(marker_lag_counts)}"
        )
    expected_lag_count = next(iter(marker_lag_counts))

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    release_global_lock = None
    if args.global_lock_file:
        global_lock_path = Path(args.global_lock_file).expanduser()
        global_lock_path.parent.mkdir(parents=True, exist_ok=True)
        release_global_lock = acquire_lock(global_lock_path, "Global HY job")
    output_lock = out_dir / ".hy_pair_summarize.lock"
    release_output_lock = acquire_lock(output_lock, "Summarize output")
    ensure_free_disk(out_dir, args.min_free_disk_gb, "output")
    ensure_free_disk(temp_dir, args.min_free_disk_gb, "temp")

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC';")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)};")
    con.execute(f"SET temp_directory = {sql_quote(str(temp_dir))};")
    con.execute(
        f"SET max_temp_directory_size = {sql_quote(args.max_temp_size)};"
    )
    con.execute(f"SET threads = {args.threads};")
    con.execute("SET preserve_insertion_order = false;")

    component_strings = [str(path) for path in component_paths]
    source = (
        f"read_parquet({sql_list(component_strings)}, union_by_name=true, "
        "hive_partitioning=true, filename=true)"
    )
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    missing = sorted(REQUIRED_COMPONENT_COLS - cols)
    if missing:
        raise RuntimeError(f"components is missing v3.2 columns: {missing}")

    file_rows = con.execute(f"""
        SELECT
            filename,
            count(*) AS n_rows,
            count(DISTINCT run_id) AS n_run_ids,
            min(run_id) AS run_id,
            count(*) FILTER (WHERE run_id IS NULL) AS null_run_ids,
            count(DISTINCT config_hash) AS n_config_hashes,
            min(config_hash) AS config_hash,
            count(*) FILTER (WHERE config_hash IS NULL) AS null_config_hashes
        FROM {source}
        GROUP BY filename
    """).fetchall()
    stats_by_path = {str(Path(row[0]).resolve()): row[1:] for row in file_rows}
    total_component_rows = 0
    for path in component_paths:
        stats = stats_by_path.get(str(path))
        if stats is None:
            raise RuntimeError(f"No rows found for component file {path}")
        (
            n_rows,
            n_run_ids,
            row_run_id,
            null_run_ids,
            n_config_hashes,
            row_config_hash,
            null_config_hashes,
        ) = stats
        marker = markers[str(path)]
        if (
            int(n_rows) != int(marker["component_rows"])
            or n_run_ids != 1
            or null_run_ids
            or row_run_id != marker["run_id"]
            or n_config_hashes != 1
            or null_config_hashes
            or row_config_hash != marker["config_hash"]
        ):
            raise RuntimeError(
                f"Component rows do not match SUCCESS marker for {path}: "
                f"rows={n_rows}, run_id={row_run_id!r}, "
                f"config_hash={row_config_hash!r}"
            )
        total_component_rows += int(n_rows)
    if total_component_rows > args.max_component_rows:
        raise RuntimeError(
            f"Input has {total_component_rows:,} rows, exceeding "
            f"--max-component-rows={args.max_component_rows:,}"
        )

    invalid_row = con.execute(f"""
        SELECT filename, run_id, x_symbol, y_symbol, window_start, lag_ms
        FROM {source}
        WHERE window_start IS NULL
           OR window_end IS NULL
           OR window_end <= window_start
           OR cov IS NULL OR NOT isfinite(cov)
           OR var_x IS NULL OR NOT isfinite(var_x) OR var_x < 0
           OR var_y IS NULL OR NOT isfinite(var_y) OR var_y < 0
           OR n_overlap IS NULL OR n_overlap < 0
           OR n_x IS NULL OR n_x < 0
           OR n_y IS NULL OR n_y < 0
           OR corr_is_diagnostic IS DISTINCT FROM true
        LIMIT 1
    """).fetchone()
    if invalid_row is not None:
        raise RuntimeError(f"Invalid component row: {invalid_row}")

    if not args.allow_cross_base_asset:
        bad_base = con.execute(f"""
            SELECT filename, x_symbol, x_base_asset, y_symbol, y_base_asset
            FROM {source}
            WHERE x_base_asset IS NULL OR trim(x_base_asset) = ''
               OR y_base_asset IS NULL OR trim(y_base_asset) = ''
               OR upper(trim(x_base_asset)) <> upper(trim(y_base_asset))
            LIMIT 1
        """).fetchone()
        if bad_base is not None:
            raise RuntimeError(f"Cross/missing base_asset component found: {bad_base}")

    duplicate = con.execute(f"""
        SELECT {', '.join(PRIMARY_KEY_COLS)}, count(*) AS n
        FROM {source}
        GROUP BY {', '.join(PRIMARY_KEY_COLS)}
        HAVING count(*) > 1
        LIMIT 1
    """).fetchone()
    if duplicate is not None:
        raise RuntimeError(
            "Duplicate pair/window/lag primary key across component shards: "
            f"{duplicate}"
        )

    reused_run = con.execute(f"""
        SELECT run_id, count(DISTINCT filename) AS n_files
        FROM {source}
        GROUP BY run_id
        HAVING count(DISTINCT filename) <> 1
        LIMIT 1
    """).fetchone()
    if reused_run is not None:
        raise RuntimeError(f"A run_id appears in multiple files: {reused_run}")

    inconsistent_cache = con.execute(f"""
        WITH identities AS (
            SELECT
                x_exchange AS exchange,
                x_market_type AS market_type,
                x_symbol AS symbol,
                x_base_asset AS base_asset,
                x_cache_layout AS cache_layout,
                x_max_interval_ms AS max_interval_ms,
                x_cache_config_version AS cache_config_version,
                x_drop_zero_returns AS drop_zero_returns,
                x_cache_config_hash AS cache_hash
            FROM {source}
            UNION ALL
            SELECT
                y_exchange, y_market_type, y_symbol, y_base_asset,
                y_cache_layout, y_max_interval_ms, y_cache_config_version,
                y_drop_zero_returns,
                y_cache_config_hash
            FROM {source}
        )
        SELECT exchange, market_type, symbol,
               count(DISTINCT base_asset) AS n_base_assets,
               count(DISTINCT cache_layout) AS n_layouts,
               count(DISTINCT max_interval_ms) AS n_max_intervals,
               count(DISTINCT cache_config_version) AS n_versions,
               count(DISTINCT drop_zero_returns) AS n_drop_zero_modes,
               count(DISTINCT cache_hash) AS n_hashes,
               count(*) FILTER (WHERE cache_hash IS NULL) AS null_hashes
        FROM identities
        GROUP BY exchange, market_type, symbol
        HAVING count(DISTINCT base_asset) <> 1
            OR count(DISTINCT cache_layout) > 1
            OR count(DISTINCT max_interval_ms) > 1
            OR count(DISTINCT cache_config_version) > 1
            OR count(DISTINCT drop_zero_returns) > 1
            OR count(DISTINCT cache_hash) <> 1
            OR count(*) FILTER (WHERE cache_hash IS NULL) > 0
        LIMIT 1
    """).fetchone()
    if inconsistent_cache is not None:
        raise RuntimeError(
            f"Cache identity changed across component shards: {inconsistent_cache}"
        )

    pair_count = int(con.execute(f"""
        SELECT count(*)
        FROM (SELECT DISTINCT {', '.join(PAIR_ID_COLS)} FROM {source})
    """).fetchone()[0])
    if effective_expected_pairs > 0 and pair_count != effective_expected_pairs:
        raise RuntimeError(
            f"Found {pair_count} distinct pairs; expected "
            f"{effective_expected_pairs}."
        )

    incomplete_lags = con.execute(f"""
        SELECT {', '.join(PAIR_COLS)}, count(DISTINCT lag_ms) AS n_lags,
               count(*) FILTER (WHERE lag_ms = 0) AS zero_rows
        FROM {source}
        GROUP BY {', '.join(PAIR_COLS)}
        HAVING count(DISTINCT lag_ms) <> {expected_lag_count}
            OR count(*) FILTER (WHERE lag_ms = 0) = 0
        LIMIT 1
    """).fetchone()
    if incomplete_lags is not None:
        raise RuntimeError(f"Incomplete lag grid for pair: {incomplete_lags}")

    uneven_windows = con.execute(f"""
        WITH per_lag AS (
            SELECT {', '.join(PAIR_COLS)}, lag_ms, count(*) AS n_windows
            FROM {source}
            GROUP BY {', '.join(PAIR_COLS)}, lag_ms
        )
        SELECT {', '.join(PAIR_COLS)}, min(n_windows), max(n_windows)
        FROM per_lag
        GROUP BY {', '.join(PAIR_COLS)}
        HAVING min(n_windows) <> max(n_windows)
        LIMIT 1
    """).fetchone()
    if uneven_windows is not None:
        raise RuntimeError(f"Lag window counts differ within pair: {uneven_windows}")

    print(
        f"[PREFLIGHT] validation passed: files={len(component_paths):,} "
        f"rows={total_component_rows:,} pairs={pair_count:,} "
        f"lags={expected_lag_count} config_hash={config_hash}"
    )

    group_keys = GROUP_COLS
    metadata_cols = CACHE_META_COLS
    lag_group_cols = group_keys + ["lag_ms"]

    group_key_select = comma(group_keys)
    lag_group_select = comma(lag_group_cols)
    lag_group_by = ", ".join(lag_group_cols)
    key_using = using_clause(group_keys)

    meta_select = ",\n        " + ",\n        ".join(
        f"min({c}) AS {c}" for c in metadata_cols
    )

    eligibility = f"""
        total_overlap >= {args.min_overlap}
        AND total_n_x >= {args.min_n_x}
        AND total_n_y >= {args.min_n_y}
        AND n_window_rows >= {args.min_window_rows}
        AND ABS(lag_ms) >= {args.min_abs_lag_ms}
        AND agg_corr IS NOT NULL
    """

    lag_summary_parquet = out_dir / "hy_pair_lag_summary.parquet"
    lag_summary_csv = out_dir / "hy_pair_lag_summary.csv"
    best_abs_parquet = out_dir / "hy_pair_best_abs_summary.parquet"
    best_abs_csv = out_dir / "hy_pair_best_abs_summary.csv"
    best_pos_parquet = out_dir / "hy_pair_best_positive_summary.parquet"
    best_pos_csv = out_dir / "hy_pair_best_positive_summary.csv"

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE lag_summary AS
    WITH aggregated AS (
        SELECT
            {lag_group_select},
            SUM(cov) AS sum_cov,
            SUM(var_x) AS sum_var_x,
            SUM(var_y) AS sum_var_y,
            SUM(n_overlap) AS total_overlap,
            SUM(n_x) AS total_n_x,
            SUM(n_y) AS total_n_y,
            COUNT(*) AS n_window_rows,
            count(DISTINCT run_id) AS n_runs,
            count(DISTINCT filename) AS n_component_files,
            min(window_start) AS analysis_window_start,
            max(window_end) AS analysis_window_end,
            SUM(CASE WHEN n_overlap > 0 THEN 1 ELSE 0 END)
                AS n_nonzero_overlap_windows,
            SUM(CASE WHEN n_overlap > 0 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*), 0) AS overlap_window_ratio
            {meta_select}
        FROM {source}
        GROUP BY {lag_group_by}
    )
    , scored AS (
        SELECT
            *,
            CASE
                WHEN isfinite(sum_cov)
                 AND isfinite(sum_var_x) AND sum_var_x > 0
                 AND isfinite(sum_var_y) AND sum_var_y > 0
                THEN sum_cov / (sqrt(sum_var_x) * sqrt(sum_var_y))
                ELSE NULL
            END AS agg_corr
        FROM aggregated
    )
    SELECT
        *,
        true AS corr_is_diagnostic,
        CASE
            WHEN agg_corr IS NULL THEN NULL
            ELSE ABS(agg_corr) > 1.0
        END AS corr_outside_unit_interval
    FROM scored;
    """)

    # Best by absolute correlation. This is useful for anomaly discovery, but a negative
    # best correlation should be interpreted carefully for same-underlying lead-lag work.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE best_abs AS
    WITH eligible AS (
        SELECT *
        FROM lag_summary
        WHERE {eligibility}
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY {', '.join(group_keys)}
                ORDER BY ABS(agg_corr) DESC NULLS LAST, total_overlap DESC, ABS(lag_ms) ASC
            ) AS rn
        FROM eligible
    ),
    zero_lag AS (
        SELECT
            {group_key_select},
            agg_corr AS corr_0_lag,
            total_overlap AS overlap_0_lag
        FROM lag_summary
        WHERE lag_ms = 0
    ),
    second_best AS (
        SELECT
            {group_key_select},
            lag_ms AS second_best_lag_ms,
            agg_corr AS second_best_agg_corr,
            total_overlap AS second_best_total_overlap
        FROM ranked
        WHERE rn = 2
    )
    SELECT
        {comma(prefixed(group_keys, 'r'))},
        r.lag_ms AS best_lag_ms,
        r.agg_corr AS best_agg_corr,
        r.corr_is_diagnostic,
        r.corr_outside_unit_interval,
        CASE WHEN r.agg_corr < 0 THEN true ELSE false END AS best_is_negative,
        z.corr_0_lag,
        z.overlap_0_lag,
        r.agg_corr - z.corr_0_lag AS best_minus_zero_corr,
        ABS(r.agg_corr) - ABS(z.corr_0_lag) AS best_abs_minus_zero_abs_corr,
        s.second_best_lag_ms,
        s.second_best_agg_corr,
        s.second_best_total_overlap,
        ABS(r.agg_corr) - ABS(COALESCE(s.second_best_agg_corr, 0.0)) AS best_abs_corr_margin,
        r.sum_cov,
        r.sum_var_x,
        r.sum_var_y,
        r.total_overlap,
        r.total_n_x,
        r.total_n_y,
        r.n_window_rows,
        r.n_nonzero_overlap_windows,
        r.overlap_window_ratio,
        r.n_runs,
        r.n_component_files,
        r.analysis_window_start,
        r.analysis_window_end
        {',' if metadata_cols else ''}
        {comma(prefixed(metadata_cols, 'r')) if metadata_cols else ''}
    FROM ranked r
    LEFT JOIN zero_lag z USING {key_using}
    LEFT JOIN second_best s USING {key_using}
    WHERE r.rn = 1
    ORDER BY ABS(r.agg_corr) DESC NULLS LAST, r.total_overlap DESC;
    """)

    # Best positive correlation. For same-symbol spot/perp or venue lead-lag, this is
    # usually the more interpretable table than best_abs.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE best_positive AS
    WITH eligible AS (
        SELECT *
        FROM lag_summary
        WHERE {eligibility}
          AND agg_corr > 0
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY {', '.join(group_keys)}
                ORDER BY agg_corr DESC NULLS LAST, total_overlap DESC, ABS(lag_ms) ASC
            ) AS rn
        FROM eligible
    ),
    zero_lag AS (
        SELECT
            {group_key_select},
            agg_corr AS corr_0_lag,
            total_overlap AS overlap_0_lag
        FROM lag_summary
        WHERE lag_ms = 0
    ),
    second_best AS (
        SELECT
            {group_key_select},
            lag_ms AS second_best_positive_lag_ms,
            agg_corr AS second_best_positive_agg_corr,
            total_overlap AS second_best_positive_total_overlap
        FROM ranked
        WHERE rn = 2
    )
    SELECT
        {comma(prefixed(group_keys, 'r'))},
        r.lag_ms AS best_positive_lag_ms,
        r.agg_corr AS best_positive_agg_corr,
        r.corr_is_diagnostic,
        r.corr_outside_unit_interval,
        z.corr_0_lag,
        z.overlap_0_lag,
        r.agg_corr - z.corr_0_lag AS best_positive_minus_zero_corr,
        s.second_best_positive_lag_ms,
        s.second_best_positive_agg_corr,
        s.second_best_positive_total_overlap,
        r.agg_corr - COALESCE(s.second_best_positive_agg_corr, 0.0) AS best_positive_corr_margin,
        r.sum_cov,
        r.sum_var_x,
        r.sum_var_y,
        r.total_overlap,
        r.total_n_x,
        r.total_n_y,
        r.n_window_rows,
        r.n_nonzero_overlap_windows,
        r.overlap_window_ratio,
        r.n_runs,
        r.n_component_files,
        r.analysis_window_start,
        r.analysis_window_end
        {',' if metadata_cols else ''}
        {comma(prefixed(metadata_cols, 'r')) if metadata_cols else ''}
    FROM ranked r
    LEFT JOIN zero_lag z USING {key_using}
    LEFT JOIN second_best s USING {key_using}
    WHERE r.rn = 1
    ORDER BY r.agg_corr DESC NULLS LAST, r.total_overlap DESC;
    """)

    lag_summary_rows = int(con.execute("SELECT count(*) FROM lag_summary").fetchone()[0])
    best_abs_rows = int(con.execute("SELECT count(*) FROM best_abs").fetchone()[0])
    best_positive_rows = int(
        con.execute("SELECT count(*) FROM best_positive").fetchone()[0]
    )
    output_tables = [
        ("lag_summary", lag_summary_parquet),
        ("lag_summary", lag_summary_csv),
        ("best_abs", best_abs_parquet),
        ("best_abs", best_abs_csv),
        ("best_positive", best_pos_parquet),
        ("best_positive", best_pos_csv),
    ]
    partial_outputs = [
        (write_table_partial(con, table, final_path), final_path)
        for table, final_path in output_tables
    ]
    ensure_free_disk(out_dir, args.min_free_disk_gb, "output")
    ensure_free_disk(temp_dir, args.min_free_disk_gb, "temp")

    success_path = out_dir / "hy_pair_summary.SUCCESS.json"
    partial_success = out_dir / "hy_pair_summary.SUCCESS.json.partial"
    if success_path.exists():
        success_path.unlink()
    for partial_path, final_path in partial_outputs:
        partial_path.replace(final_path)

    partial_success.write_text(
        json.dumps(
            {
                "status": "complete",
                "script_version": SCRIPT_VERSION,
                "duckdb_version": duckdb.__version__,
                "completed_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "components_pattern": args.components,
                "component_files": [str(path) for path in component_paths],
                "component_run_ids": sorted(
                    str(marker["run_id"]) for marker in markers.values()
                ),
                "config_hash": config_hash,
                "input_component_rows": total_component_rows,
                "input_component_files": len(component_paths),
                "distinct_pairs": pair_count,
                "expected_pairs": effective_expected_pairs,
                "pairs_csv": (
                    str(Path(args.pairs_csv).expanduser().resolve())
                    if args.pairs_csv
                    else None
                ),
                "expected_lag_count": expected_lag_count,
                "lag_summary_rows": lag_summary_rows,
                "best_abs_rows": best_abs_rows,
                "best_positive_rows": best_positive_rows,
                "thresholds": {
                    "min_overlap": args.min_overlap,
                    "min_n_x": args.min_n_x,
                    "min_n_y": args.min_n_y,
                    "min_window_rows": args.min_window_rows,
                    "min_abs_lag_ms": args.min_abs_lag_ms,
                },
                "require_equal_base_asset": not args.allow_cross_base_asset,
                "primary_key": PRIMARY_KEY_COLS,
                "outputs": [str(path) for _, path in output_tables],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    partial_success.replace(success_path)
    con.close()
    release_output_lock()
    atexit.unregister(release_output_lock)
    if release_global_lock is not None:
        release_global_lock()
        atexit.unregister(release_global_lock)

    print(f"[DONE] wrote {lag_summary_parquet}")
    print(f"[DONE] wrote {lag_summary_csv}")
    print(f"[DONE] wrote {best_abs_parquet}")
    print(f"[DONE] wrote {best_abs_csv}")
    print(f"[DONE] wrote {best_pos_parquet}")
    print(f"[DONE] wrote {best_pos_csv}")
    print(f"[DONE] wrote completion marker: {success_path}")
    print(
        f"[DONE] pairs={pair_count} input_rows={total_component_rows:,} "
        f"config_hash={config_hash}"
    )
    print(
        f"[DONE] DuckDB memory_limit={args.memory_limit} "
        f"temp_dir={temp_dir} threads={args.threads}"
    )


if __name__ == "__main__":
    main()
