#!/usr/bin/env python3
"""Scan hourly BBO Parquet files and write one statistics row per file.

Receive time can use the low-RAM physical-order path. Event and transaction
time can contain repeated timestamps, so the keep-last path deterministically
selects the final BBO update at each timestamp before computing gaps/returns.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from common import (
    choose_timestamp_basis_col,
    load_excluded_hours,
    order_cols_for_dedup,
    resolve_excluded_hours_path,
)


SCRIPT_VERSION = "2.2.0"

PUBLIC_COLUMNS = [
    "exchange",
    "market_type",
    "symbol",
    "parquet_glob",
    "timestamp_basis",
    "timestamp_col",
    "duplicate_ts_policy",
    "scan_config_hash",
    "scan_ts_unit",
    "scan_price_mode",
    "scan_max_spread_bps",
    "scan_start",
    "scan_end",
    "file_path",
    "total_rows",
    "selected_rows",
    "deduplicated_rows",
    "clean_rows",
    "first_ts",
    "last_ts",
    "duplicate_ts",
    "backward_ts",
    "crossed_rows",
    "locked_rows",
    "non_positive_bid",
    "non_positive_ask",
    "non_positive_bid_qty",
    "non_positive_ask_qty",
    "gap_p50_ms",
    "gap_p90_ms",
    "gap_p95_ms",
    "gap_p99_ms",
    "gap_p999_ms",
    "gap_max_ms",
    "zero_ret_count",
    "ret_count",
    "spread_bps_p50",
    "spread_bps_p99",
]

# These small state fields make --resume and cross-hour intervals exact.  They
# are ignored by the second-stage aggregation program.
STATE_COLUMNS = [
    "file_max_valid_ts",
    "file_last_input_ts",
    "file_last_clean_ts",
    "file_last_clean_price",
    "processing_mode",
]
PARTIAL_COLUMNS = PUBLIC_COLUMNS + STATE_COLUMNS
ResumeKey = tuple[str, str, str, str]


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def scan_expr(path: str) -> str:
    return (
        f"read_parquet({sql_quote(path)}, "
        "union_by_name = true, hive_partitioning = true)"
    )


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def ts_literal(value: datetime | None) -> str:
    if value is None:
        return "CAST(NULL AS TIMESTAMP)"
    value = parse_dt(value.isoformat())
    assert value is not None
    return f"TIMESTAMP {sql_quote(value.isoformat(sep=' '))}"


def double_literal(value: float | None) -> str:
    if value is None:
        return "CAST(NULL AS DOUBLE)"
    if not math.isfinite(value):
        raise ValueError(f"non-finite boundary price: {value!r}")
    return f"CAST({value!r} AS DOUBLE)"


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def read_universe(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"exchange", "market_type", "symbol", "parquet_glob"}
    if not rows:
        raise ValueError(f"empty universe CSV: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"universe CSV missing columns: {sorted(missing)}")
    return rows


def hour_from_path(path: str) -> datetime | None:
    date = None
    hour = None
    for part in Path(path).parts:
        if part.startswith("date="):
            date = part.split("=", 1)[1]
        elif part.startswith("hour="):
            hour = part.split("=", 1)[1]
    if date is None or hour is None:
        return None
    try:
        return datetime.fromisoformat(f"{date} {hour}:00:00")
    except ValueError:
        return None


def file_sort_key(path: str) -> tuple[bool, datetime, str]:
    hour = hour_from_path(path)
    return (hour is None, hour or datetime.max, path)


def file_overlaps_range(
    path: str,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    hour_start = hour_from_path(path)
    if hour_start is None:
        return True
    hour_end = hour_start + timedelta(hours=1)
    return not (
        (start is not None and hour_end <= start)
        or (end is not None and hour_start >= end)
    )


def get_columns(con: duckdb.DuckDBPyConnection, path: str) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {scan_expr(path)}").fetchall()
    return [str(row[0]) for row in rows]


def build_ts_expr(column: str, unit: str) -> str:
    name = quote_identifier(column)
    if unit == "timestamp":
        return f"TRY_CAST({name} AS TIMESTAMP)"
    if unit == "ns":
        return f"make_timestamp_ns(TRY_CAST({name} AS BIGINT))"
    if unit == "us":
        return f"make_timestamp(TRY_CAST({name} AS BIGINT))"
    if unit == "ms":
        return (
            "TIMESTAMP '1970-01-01' + "
            f"TRY_CAST({name} AS BIGINT) * INTERVAL 1 MILLISECOND"
        )
    raise ValueError(f"unsupported timestamp unit: {unit}")


def connect(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(str(args.temp_dir))}")
    con.execute(
        f"SET max_temp_directory_size = {sql_quote(args.max_temp_size)}"
    )
    con.execute(f"SET threads = {args.threads}")
    # The fast path intentionally uses physical Parquet row order.
    con.execute("SET preserve_insertion_order = true")
    con.execute("SET enable_object_cache = false")
    return con


def price_sql(price_mode: str) -> str:
    if price_mode == "mid":
        return "(_bid_price + _ask_price) / 2.0"
    return (
        "(_bid_price * _ask_qty + _ask_price * _bid_qty) "
        "/ (_bid_qty + _ask_qty)"
    )


def common_ctes(
    file_path: str,
    symbol: str,
    ts_expr: str,
    time_filter: str,
    price_mode: str,
    max_spread_bps: float,
    dedup_order_by: str,
    duplicate_ts_policy: str,
) -> str:
    price = price_sql(price_mode)
    if duplicate_ts_policy == "keep-last":
        selection_ctes = f"""
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY ts ORDER BY {dedup_order_by}
                ) AS _dedup_rank
            FROM marked
            WHERE ts IS NOT NULL AND isfinite(ts)
        ),
        selected AS (
            SELECT * EXCLUDE (_dedup_rank)
            FROM ranked
            WHERE _dedup_rank = 1
        )
        """
    elif duplicate_ts_policy == "error":
        selection_ctes = """
        selected AS (
            SELECT *
            FROM marked
            WHERE ts IS NOT NULL AND isfinite(ts)
        )
        """
    else:
        raise ValueError(
            f"unsupported duplicate timestamp policy: {duplicate_ts_policy}"
        )
    return f"""
        raw AS (
            SELECT
                *,
                {ts_expr} AS ts,
                TRY_CAST(bid_price AS DOUBLE) AS _bid_price,
                TRY_CAST(ask_price AS DOUBLE) AS _ask_price,
                TRY_CAST(bid_qty AS DOUBLE) AS _bid_qty,
                TRY_CAST(ask_qty AS DOUBLE) AS _ask_qty
            FROM {scan_expr(file_path)}
            WHERE CAST(symbol AS VARCHAR) = {sql_quote(symbol)}
        ),
        filtered AS (
            SELECT * FROM raw WHERE {time_filter}
        ),
        enriched AS (
            SELECT
                *,
                CASE
                    WHEN _bid_price > 0 AND _ask_price > 0
                     AND isfinite(_bid_price) AND isfinite(_ask_price)
                    THEN 10000.0 * (_ask_price - _bid_price)
                         / ((_bid_price + _ask_price) / 2.0)
                END AS spread_bps,
                CASE
                    WHEN _bid_price > 0 AND _ask_price > 0
                     AND _bid_qty > 0 AND _ask_qty > 0
                     AND isfinite(_bid_price) AND isfinite(_ask_price)
                     AND isfinite(_bid_qty) AND isfinite(_ask_qty)
                    THEN {price}
                END AS price
            FROM filtered
        ),
        marked AS (
            SELECT
                *,
                ts IS NOT NULL
                AND _bid_price > 0 AND _ask_price > 0
                AND _bid_qty > 0 AND _ask_qty > 0
                AND isfinite(_bid_price) AND isfinite(_ask_price)
                AND isfinite(_bid_qty) AND isfinite(_ask_qty)
                AND _ask_price > _bid_price
                AND spread_bps BETWEEN 0 AND {max_spread_bps!r}
                AND price > 0 AND isfinite(price) AS is_clean,
                _bid_price > _ask_price AS is_crossed,
                _bid_price = _ask_price AND _bid_price > 0 AS is_locked,
                _bid_price <= 0 AS has_non_positive_bid,
                _ask_price <= 0 AS has_non_positive_ask,
                _bid_qty <= 0 AS has_non_positive_bid_qty,
                _ask_qty <= 0 AS has_non_positive_ask_qty
            FROM enriched
        ),
        {selection_ctes}
    """


def scan_base_stats(
    con: duckdb.DuckDBPyConnection,
    row: dict[str, str],
    file_path: str,
    ts_expr: str,
    time_filter: str,
    price_mode: str,
    max_spread_bps: float,
    previous_valid_ts: datetime | None,
    previous_input_ts: datetime | None,
    dedup_order_by: str,
    duplicate_ts_policy: str,
) -> dict[str, Any]:
    """Calculate base quality fields and verify the ascending-time contract."""
    ctes = common_ctes(
        file_path,
        row["symbol"],
        ts_expr,
        time_filter,
        price_mode,
        max_spread_bps,
        dedup_order_by,
        duplicate_ts_policy,
    )
    query = f"""
        WITH {ctes},
        input_window AS (
            SELECT *, LAG(ts) OVER () AS file_prev_ts
            FROM marked
        ),
        raw_agg AS (
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE ts IS NOT NULL AND isfinite(ts)
                ) AS valid_timestamp_rows,
                COUNT(*) FILTER (
                    WHERE ts = COALESCE(
                        file_prev_ts, {ts_literal(previous_valid_ts)}
                    )
                ) AS duplicate_ts,
                COUNT(*) FILTER (
                    WHERE ts < COALESCE(
                        file_prev_ts, {ts_literal(previous_input_ts)}
                    )
                ) AS backward_ts,
                COUNT(*) FILTER (WHERE is_crossed) AS crossed_rows,
                COUNT(*) FILTER (WHERE is_locked) AS locked_rows,
                COUNT(*) FILTER (
                    WHERE has_non_positive_bid
                ) AS non_positive_bid,
                COUNT(*) FILTER (
                    WHERE has_non_positive_ask
                ) AS non_positive_ask,
                COUNT(*) FILTER (
                    WHERE has_non_positive_bid_qty
                ) AS non_positive_bid_qty,
                COUNT(*) FILTER (
                    WHERE has_non_positive_ask_qty
                ) AS non_positive_ask_qty,
                MAX(ts) FILTER (WHERE ts IS NOT NULL) AS file_max_valid_ts,
                MAX(ts) FILTER (WHERE ts IS NOT NULL) AS file_last_input_ts
            FROM input_window
        ),
        selected_agg AS (
            SELECT
                COUNT(*) AS selected_rows,
                COUNT(*) FILTER (WHERE is_clean) AS clean_rows,
                MIN(ts) AS first_ts,
                MAX(ts) AS last_ts,
                approx_quantile(
                    spread_bps, [0.50, 0.99]::FLOAT[]
                ) FILTER (
                    WHERE spread_bps >= 0 AND isfinite(spread_bps)
                ) AS spread_q,
                MAX(ts) FILTER (WHERE is_clean) AS file_last_clean_ts,
                arg_max(price, ts) FILTER (
                    WHERE is_clean
                ) AS file_last_clean_price
            FROM selected
        )
        SELECT
            raw_agg.total_rows,
            raw_agg.valid_timestamp_rows,
            selected_agg.selected_rows,
            selected_agg.clean_rows,
            selected_agg.first_ts,
            selected_agg.last_ts,
            raw_agg.duplicate_ts AS physical_duplicate_ts,
            backward_ts,
            crossed_rows, locked_rows,
            non_positive_bid, non_positive_ask,
            non_positive_bid_qty, non_positive_ask_qty,
            selected_agg.spread_q[1] AS spread_bps_p50,
            selected_agg.spread_q[2] AS spread_bps_p99,
            file_max_valid_ts, file_last_input_ts,
            file_last_clean_ts, file_last_clean_price
        FROM raw_agg CROSS JOIN selected_agg
    """
    cursor = con.execute(query)
    names = [item[0] for item in cursor.description]
    values = cursor.fetchone()
    if values is None:
        raise RuntimeError(f"base statistics returned no row: {file_path}")
    result = dict(zip(names, values))
    local_deduplicated = max(
        0,
        int(result.pop("valid_timestamp_rows"))
        - int(result["selected_rows"]),
    )
    boundary_duplicate = int(
        previous_valid_ts is not None
        and result["first_ts"] == previous_valid_ts
    )
    if duplicate_ts_policy == "keep-last":
        result["duplicate_ts"] = local_deduplicated + boundary_duplicate
        result["deduplicated_rows"] = result["duplicate_ts"]
    else:
        result["duplicate_ts"] = int(result.pop("physical_duplicate_ts"))
        result["deduplicated_rows"] = result["duplicate_ts"]
    result.pop("physical_duplicate_ts", None)
    if duplicate_ts_policy == "error" and result["backward_ts"] > 0:
        raise RuntimeError(
            f"selected timestamp is not ascending in {file_path}: "
            f"backward_ts={result['backward_ts']}"
        )
    if duplicate_ts_policy == "error" and result["duplicate_ts"] > 0:
        raise RuntimeError(
            f"duplicate selected timestamps in {file_path}: "
            f"duplicate_ts={result['duplicate_ts']}; use "
            "--duplicate-ts-policy keep-last for event/transaction time"
        )
    return result


def scan_clean_time_stats(
    con: duckdb.DuckDBPyConnection,
    row: dict[str, str],
    file_path: str,
    ts_expr: str,
    time_filter: str,
    price_mode: str,
    max_spread_bps: float,
    previous_clean_ts: datetime | None,
    previous_clean_price: float | None,
    dedup_order_by: str,
    duplicate_ts_policy: str,
) -> dict[str, Any]:
    """Calculate clean gaps and returns in a second, narrow streaming pass."""
    ctes = common_ctes(
        file_path,
        row["symbol"],
        ts_expr,
        time_filter,
        price_mode,
        max_spread_bps,
        dedup_order_by,
        duplicate_ts_policy,
    )
    query = f"""
        WITH {ctes},
        clean AS (
            SELECT ts, price
            FROM selected
            WHERE is_clean
        ),
        clean_window AS (
            SELECT
                ts,
                price,
                LAG(ts) OVER (ORDER BY ts) AS file_prev_ts,
                LAG(price) OVER (ORDER BY ts) AS file_prev_price
            FROM clean
        ),
        metrics AS (
            SELECT
                ts,
                price,
                COALESCE(
                    file_prev_price, {double_literal(previous_clean_price)}
                ) AS prev_price,
                datediff(
                    'microsecond',
                    COALESCE(file_prev_ts, {ts_literal(previous_clean_ts)}),
                    ts
                ) / 1000.0 AS interval_ms
            FROM clean_window
        ),
        agg AS (
            SELECT
                approx_quantile(
                    interval_ms,
                    [0.50, 0.90, 0.95, 0.99, 0.999]::FLOAT[]
                ) FILTER (WHERE interval_ms > 0) AS gap_q,
                MAX(interval_ms) FILTER (
                    WHERE interval_ms > 0
                ) AS gap_max_ms,
                COUNT(*) FILTER (
                    WHERE interval_ms > 0
                      AND prev_price > 0
                      AND price = prev_price
                ) AS zero_ret_count,
                COUNT(*) FILTER (
                    WHERE interval_ms > 0 AND prev_price > 0
                ) AS ret_count
            FROM metrics
        )
        SELECT
            gap_q[1] AS gap_p50_ms,
            gap_q[2] AS gap_p90_ms,
            gap_q[3] AS gap_p95_ms,
            gap_q[4] AS gap_p99_ms,
            gap_q[5] AS gap_p999_ms,
            gap_max_ms,
            zero_ret_count,
            ret_count
        FROM agg
    """
    cursor = con.execute(query)
    names = [item[0] for item in cursor.description]
    values = cursor.fetchone()
    if values is None:
        raise RuntimeError(f"time statistics returned no row: {file_path}")
    return dict(zip(names, values))


def validate_file_columns(
    columns: list[str],
    file_path: str,
    timestamp_col: str,
) -> None:
    available = set(columns)
    required = {
        "symbol",
        timestamp_col,
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{file_path}: missing columns: {', '.join(missing)}")


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def resume_key(row: dict[str, str], file_path: str | None = None) -> ResumeKey:
    return (
        row["exchange"],
        row["market_type"],
        row["symbol"],
        file_path if file_path is not None else row["file_path"],
    )


def load_resume_rows(
    path: Path,
) -> tuple[dict[ResumeKey, dict[str, str]], set[ResumeKey]]:
    if not path.exists():
        return {}, set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PARTIAL_COLUMNS:
            raise ValueError(
                "existing partial CSV has a different schema; "
                "use another --partial-out or remove it"
            )
        rows: dict[ResumeKey, dict[str, str]] = {}
        for row in reader:
            key = resume_key(row)
            if key in rows:
                raise ValueError(
                    "duplicate instrument/file row in partial CSV: "
                    f"{key!r}"
                )
            rows[key] = row
    return rows, set(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-RAM per-hour quality statistics (stage 1 of 2)."
    )
    parser.add_argument("--universe", required=True)
    parser.add_argument(
        "--excluded-hours",
        help=(
            "Optional abnormal-hour CSV. By default excluded_hours.csv beside "
            "the universe is used when present."
        ),
    )
    parser.add_argument("--partial-out", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--timestamp-col")
    parser.add_argument(
        "--timestamp-basis",
        choices=["receive", "event", "transaction"],
        help=(
            "Clock used by HY. Defaults to receive unless --timestamp-col is "
            "provided, in which case the basis is recorded as custom."
        ),
    )
    parser.add_argument(
        "--duplicate-ts-policy",
        choices=["error", "keep-last"],
        default="error",
        help=(
            "How to handle repeated selected timestamps. Event/transaction "
            "runs should use keep-last."
        ),
    )
    parser.add_argument(
        "--missing-timestamp-policy",
        choices=["error", "skip"],
        default="error",
        help=(
            "Use skip for transaction-time runs so Binance Spot, whose schema "
            "has no transaction_time, is excluded explicitly."
        ),
    )
    parser.add_argument(
        "--ts-unit",
        choices=["timestamp", "ns", "us", "ms"],
        default="timestamp",
    )
    parser.add_argument(
        "--price-mode", choices=["mid", "microprice"], default="mid"
    )
    parser.add_argument("--max-spread-bps", type=float, default=100.0)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--temp-dir", default="./duckdb_tmp")
    parser.add_argument("--max-temp-size", default="8GB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--reopen-every",
        type=int,
        default=10,
        help="Reopen DuckDB after this many processed files; 0 disables it.",
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=3.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing partial CSV and skip completed rows.",
    )
    args = parser.parse_args()

    if args.timestamp_col and args.timestamp_basis:
        parser.error("use either --timestamp-col or --timestamp-basis, not both")
    timestamp_basis = (
        "custom" if args.timestamp_col else (args.timestamp_basis or "receive")
    )

    if args.threads != 1:
        parser.error(
            "--threads must be 1 because statistics rely on physical row order"
        )
    if args.reopen_every < 0:
        parser.error("--reopen-every must be >= 0")
    if not math.isfinite(args.max_spread_bps) or args.max_spread_bps <= 0:
        parser.error("--max-spread-bps must be finite and > 0")
    if not math.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb must be finite and >= 0")

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    if start is not None and end is not None and start >= end:
        parser.error("--start must be earlier than --end")

    universe_path = Path(args.universe).expanduser().resolve()
    excluded_path = resolve_excluded_hours_path(
        universe_path, args.excluded_hours
    )
    excluded_by_key = load_excluded_hours(excluded_path)
    excluded_count = sum(len(hours) for hours in excluded_by_key.values())
    if excluded_path is not None:
        print(
            f"[INFO] excluded abnormal hours={excluded_count} "
            f"manifest={excluded_path}"
        )

    args.temp_dir = Path(args.temp_dir).expanduser().resolve()
    partial_out = Path(args.partial_out).expanduser().resolve()
    config_path = partial_out.with_name(partial_out.name + ".CONFIG.json")
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    partial_out.parent.mkdir(parents=True, exist_ok=True)

    scan_config = {
        "script_version": SCRIPT_VERSION,
        "universe_path": str(universe_path),
        "universe_sha256": sha256_file(universe_path),
        "excluded_hours_path": str(excluded_path) if excluded_path else None,
        "excluded_hours_sha256": sha256_file(excluded_path),
        "start": start.isoformat(timespec="microseconds") if start else None,
        "end": end.isoformat(timespec="microseconds") if end else None,
        "timestamp_basis": timestamp_basis,
        "timestamp_col": args.timestamp_col,
        "duplicate_ts_policy": args.duplicate_ts_policy,
        "missing_timestamp_policy": args.missing_timestamp_policy,
        "ts_unit": args.ts_unit,
        "price_mode": args.price_mode,
        "max_spread_bps": args.max_spread_bps,
        "processing_mode": (
            "ordered_keep_last"
            if args.duplicate_ts_policy == "keep-last"
            else "streaming_ascending_unique"
        ),
    }
    scan_config_hash = stable_config_hash(scan_config)
    config_document = {
        "scan_config_hash": scan_config_hash,
        "scan_config": scan_config,
    }

    if args.resume and partial_out.exists():
        if not config_path.is_file():
            raise RuntimeError(
                f"existing partial CSV has no configuration manifest: {config_path}; "
                "restart without --resume so incompatible scans cannot be mixed"
            )
        try:
            existing_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read scan config {config_path}: {exc}") from exc
        if existing_config != config_document:
            raise RuntimeError(
                f"resume configuration mismatch for {partial_out}; use a new "
                "--partial-out or restart without --resume"
            )
    else:
        write_json_atomic(config_path, config_document)

    time_filter = "TRUE"
    if start is not None:
        time_filter += f" AND ts >= {ts_literal(start)}"
    if end is not None:
        time_filter += f" AND ts < {ts_literal(end)}"

    resume_rows: dict[ResumeKey, dict[str, str]] = {}
    completed: set[ResumeKey] = set()
    if args.resume:
        resume_rows, completed = load_resume_rows(partial_out)
        stale_excluded_rows = [
            row
            for row in resume_rows.values()
            if hour_from_path(row["file_path"])
            in excluded_by_key.get(
                (row["exchange"], row["market_type"], row["symbol"]),
                set(),
            )
        ]
        if stale_excluded_rows:
            raise RuntimeError(
                "existing partial CSV contains rows from hours now excluded as "
                "abnormal; rebuild --partial-out without --resume"
            )

    mode = "a" if args.resume and partial_out.exists() else "w"
    rows = read_universe(universe_path)
    con = connect(args)
    processed_since_reopen = 0
    total_processed = 0

    try:
        with partial_out.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PARTIAL_COLUMNS)
            if mode == "w":
                writer.writeheader()

            for i, row in enumerate(rows, 1):
                universe_basis = row.get("timestamp_basis")
                if universe_basis and universe_basis != timestamp_basis:
                    raise RuntimeError(
                        f"universe timestamp_basis={universe_basis!r} for "
                        f"{row['exchange']}/{row['market_type']}/"
                        f"{row['symbol']}, but quality scan requested "
                        f"{timestamp_basis!r}"
                    )
                excluded_hours = excluded_by_key.get(
                    (row["exchange"], row["market_type"], row["symbol"]),
                    set(),
                )
                files: list[str] = []
                skipped_excluded_files = 0
                for path in glob.iglob(row["parquet_glob"]):
                    if not file_overlaps_range(path, start, end):
                        continue
                    if hour_from_path(path) in excluded_hours:
                        skipped_excluded_files += 1
                        continue
                    files.append(path)
                files.sort(key=file_sort_key)
                if skipped_excluded_files:
                    print(
                        f"[INFO] {i}/{len(rows)} skipped "
                        f"{skipped_excluded_files} files from abnormal hours for "
                        f"{row['exchange']}/{row['market_type']}/{row['symbol']}"
                    )
                if not files:
                    print(
                        f"[WARN] {i}/{len(rows)} no files: "
                        f"{row['exchange']}/{row['market_type']}/{row['symbol']}"
                    )
                    continue

                columns = get_columns(con, files[0])
                try:
                    timestamp_col = choose_timestamp_basis_col(
                        set(columns), timestamp_basis, args.timestamp_col
                    )
                except RuntimeError as exc:
                    if args.missing_timestamp_policy == "skip":
                        print(
                            f"[WARN] {i}/{len(rows)} skip "
                            f"{row['exchange']}/{row['market_type']}/"
                            f"{row['symbol']}: {exc}"
                        )
                        continue
                    raise
                universe_timestamp_col = row.get("timestamp_col")
                if (
                    universe_timestamp_col
                    and timestamp_col != universe_timestamp_col
                ):
                    raise RuntimeError(
                        f"universe timestamp_col={universe_timestamp_col!r} "
                        f"but schema resolved {timestamp_col!r} for "
                        f"{row['exchange']}/{row['market_type']}/"
                        f"{row['symbol']}"
                    )
                # Compacted hourly files for one instrument share a schema, so
                # inspect it once instead of issuing DESCRIBE for every hour.
                validate_file_columns(columns, files[0], timestamp_col)
                ts_expr = build_ts_expr(timestamp_col, args.ts_unit)
                dedup_order_by = order_cols_for_dedup(set(columns))
                print(
                    f"[INFO] {i}/{len(rows)} "
                    f"{row['exchange']}/{row['market_type']}/{row['symbol']} "
                    f"files={len(files)} basis={timestamp_basis} "
                    f"ts={timestamp_col} dedup={args.duplicate_ts_policy}"
                )

                previous_valid_ts = None
                previous_input_ts = None
                previous_clean_ts = None
                previous_clean_price = None
                previous_file_hour = None

                for j, file_path in enumerate(files, 1):
                    current_file_hour = hour_from_path(file_path)
                    if (
                        previous_file_hour is not None
                        and current_file_hour is not None
                        and any(
                            previous_file_hour < excluded_hour < current_file_hour
                            for excluded_hour in excluded_hours
                        )
                    ):
                        # Do not treat the two sides of an intentionally removed
                        # source hour as consecutive observations.
                        previous_valid_ts = None
                        previous_input_ts = None
                        previous_clean_ts = None
                        previous_clean_price = None
                    key = resume_key(row, file_path)
                    if key in completed:
                        old = resume_rows[key]
                        previous_valid_ts = parse_dt(
                            old["file_max_valid_ts"]
                        )
                        previous_input_ts = parse_dt(
                            old["file_last_input_ts"]
                        )
                        previous_clean_ts = parse_dt(
                            old["file_last_clean_ts"]
                        )
                        previous_clean_price = parse_optional_float(
                            old["file_last_clean_price"]
                        )
                        previous_file_hour = current_file_hour
                        continue

                    if j == 1 or j % 50 == 0 or j == len(files):
                        print(f"[INFO]   file {j}/{len(files)}")

                    free_gb = shutil.disk_usage(args.temp_dir).free / 1024**3
                    if free_gb < args.min_free_disk_gb:
                        raise RuntimeError(
                            f"temp volume has only {free_gb:.2f} GiB free"
                        )

                    base = scan_base_stats(
                        con,
                        row,
                        file_path,
                        ts_expr,
                        time_filter,
                        args.price_mode,
                        args.max_spread_bps,
                        previous_valid_ts,
                        previous_input_ts,
                        dedup_order_by,
                        args.duplicate_ts_policy,
                    )
                    timing = scan_clean_time_stats(
                        con,
                        row,
                        file_path,
                        ts_expr,
                        time_filter,
                        args.price_mode,
                        args.max_spread_bps,
                        previous_clean_ts,
                        previous_clean_price,
                        dedup_order_by,
                        args.duplicate_ts_policy,
                    )

                    result = {
                        "exchange": row["exchange"],
                        "market_type": row["market_type"],
                        "symbol": row["symbol"],
                        "parquet_glob": row["parquet_glob"],
                        "timestamp_basis": timestamp_basis,
                        "timestamp_col": timestamp_col,
                        "duplicate_ts_policy": args.duplicate_ts_policy,
                        "scan_config_hash": scan_config_hash,
                        "scan_ts_unit": args.ts_unit,
                        "scan_price_mode": args.price_mode,
                        "scan_max_spread_bps": args.max_spread_bps,
                        "scan_start": scan_config["start"],
                        "scan_end": scan_config["end"],
                        "file_path": file_path,
                        **base,
                        **timing,
                        "processing_mode": scan_config["processing_mode"],
                    }
                    writer.writerow({key: result.get(key) for key in PARTIAL_COLUMNS})
                    handle.flush()

                    if base["file_max_valid_ts"] is not None:
                        previous_valid_ts = base["file_max_valid_ts"]
                    if base["file_last_input_ts"] is not None:
                        previous_input_ts = base["file_last_input_ts"]
                    if base["file_last_clean_ts"] is not None:
                        previous_clean_ts = base["file_last_clean_ts"]
                        previous_clean_price = base["file_last_clean_price"]
                    previous_file_hour = current_file_hour
                    completed.add(key)
                    total_processed += 1
                    processed_since_reopen += 1

                    if (
                        args.reopen_every > 0
                        and processed_since_reopen >= args.reopen_every
                    ):
                        con.close()
                        con = connect(args)
                        processed_since_reopen = 0
    finally:
        con.close()

    print(f"[DONE] processed new files: {total_processed}")
    print(f"[DONE] wrote partial statistics: {partial_out}")
    print(
        "[DONE] timestamp basis="
        f"{timestamp_basis} duplicate policy={args.duplicate_ts_policy}"
    )
    print("[NEXT] run 03_aggregate_quality.py on this partial CSV")


if __name__ == "__main__":
    main()
