#!/usr/bin/env python3
"""Audit local receive-time latency relative to exchange event time.

The program scans one compacted hourly Parquet file at a time so that the
largest source file, rather than the full date range, determines peak memory.
It is tailored to the compacted Binance perpetual, Binance spot, and Bybit
perpetual BBO schemas used by this project:

* Binance perpetual: ``recv_time_us`` and ``event_time``
* Binance spot: ``recv_time_ns`` and ``event_time``
* Bybit perpetual: ``recv_time_ns`` and ``event_time``

Despite their suffixes, both receive columns are TIMESTAMP WITH TIME ZONE in
the compacted Parquet schema.  Latency is therefore calculated directly as
``receive timestamp - event timestamp`` in milliseconds.
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
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import duckdb

from common import (
    ident,
    load_excluded_hours,
    parse_dt,
    read_csv_rows,
    resolve_excluded_hours_path,
    sql_quote,
)


SCRIPT_VERSION = "1.1.0"
DEFAULT_VENUES = ("binance/perp", "binance/spot", "bybit/perp")
COMPARISON_VENUE_PAIRS = (
    ("bybit/perp - binance/perp", ("bybit", "perp"), ("binance", "perp")),
    ("bybit/perp - binance/spot", ("bybit", "perp"), ("binance", "spot")),
    ("binance/perp - binance/spot", ("binance", "perp"), ("binance", "spot")),
)
QUANTILE_LEVELS = (0.01, 0.50, 0.90, 0.95, 0.99, 0.999)

COUNT_COLUMNS = [
    "total_rows",
    "valid_latency_rows",
    "recv_null_or_invalid_rows",
    "event_null_or_invalid_rows",
    "negative_latency_rows",
    "zero_latency_rows",
    "latency_gt_10ms_rows",
    "latency_gt_50ms_rows",
    "latency_gt_100ms_rows",
    "latency_gt_250ms_rows",
    "latency_gt_500ms_rows",
    "latency_gt_1000ms_rows",
    "latency_gt_5000ms_rows",
]

HOURLY_COLUMNS = [
    "exchange",
    "market_type",
    "symbol",
    "base_asset",
    "date",
    "hour",
    "file_path",
    "receive_column",
    "event_column",
    "first_receive_ts",
    "last_receive_ts",
    "first_event_ts",
    "last_event_ts",
    *COUNT_COLUMNS,
    "negative_latency_ratio",
    "latency_gt_100ms_ratio",
    "latency_min_ms",
    "latency_p01_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_p999_ms",
    "latency_mean_ms",
    "latency_stddev_ms",
    "latency_max_ms",
]

SUMMARY_METRIC_COLUMNS = [
    "n_hours",
    *COUNT_COLUMNS,
    "valid_latency_ratio",
    "negative_latency_ratio",
    "latency_gt_100ms_ratio",
    "event_weighted_mean_ms",
    "median_hour_p01_ms",
    "median_hour_p50_ms",
    "p10_hour_p50_ms",
    "p90_hour_p50_ms",
    "median_hour_p90_ms",
    "median_hour_p95_ms",
    "median_hour_p99_ms",
    "median_hour_p999_ms",
    "min_hour_p50_ms",
    "max_hour_p50_ms",
]


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def atomic_write_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    os.replace(partial, path)


def hour_from_path(path: str | Path) -> datetime | None:
    date_text = None
    hour_text = None
    for part in Path(path).parts:
        if part.startswith("date="):
            date_text = part.removeprefix("date=")
        elif part.startswith("hour="):
            hour_text = part.removeprefix("hour=")
    if date_text is None or hour_text is None:
        return None
    try:
        hour = int(hour_text)
        if not 0 <= hour <= 23:
            return None
        return datetime.fromisoformat(f"{date_text} {hour:02d}:00:00")
    except ValueError:
        return None


def file_overlaps_range(
    path: str | Path,
    start: datetime,
    end: datetime,
) -> bool:
    hour_start = hour_from_path(path)
    if hour_start is None:
        return True
    return hour_start + timedelta(hours=1) > start and hour_start < end


def timestamp_literal(value: datetime) -> str:
    # parse_dt() returns naive UTC.  Attach the offset explicitly so that range
    # filtering remains correct even if a caller supplies a DuckDB connection
    # whose session TimeZone is not UTC.
    return f"TIMESTAMPTZ {sql_quote(value.isoformat(sep=' ') + '+00:00')}"


def scan_expr(path: str | Path) -> str:
    return (
        f"read_parquet({sql_quote(str(path))}, "
        "union_by_name=true, hive_partitioning=true)"
    )


def columns_by_name(con: duckdb.DuckDBPyConnection, path: str | Path) -> dict[str, str]:
    return {
        str(name): str(column_type)
        for name, column_type, *_ in con.execute(
            f"DESCRIBE SELECT * FROM {scan_expr(path)}"
        ).fetchall()
    }


def choose_receive_column(columns: set[str]) -> str:
    for column in ("recv_time_us", "recv_time_ns", "recv_time"):
        if column in columns:
            return column
    raise RuntimeError(
        "No receive timestamp column found; expected one of "
        "recv_time_us, recv_time_ns, recv_time"
    )


def parse_venues(value: str) -> tuple[tuple[str, str], ...]:
    venues: set[tuple[str, str]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid venue {item!r}; expected exchange/market_type")
        venues.add((parts[0], parts[1]))
    return tuple(sorted(venues))


def ensure_timestamp_like(
    column: str,
    column_type: str,
    path: str | Path,
) -> None:
    if "TIMESTAMP" not in column_type.upper():
        raise RuntimeError(
            f"{column} in {path} has type {column_type!r}; this audit expects "
            "the compacted TIMESTAMP/TIMESTAMPTZ schema"
        )


def connect(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(str(args.temp_dir))}")
    con.execute(f"SET max_temp_directory_size = {sql_quote(args.max_temp_size)}")
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET preserve_insertion_order = false")
    return con


def scan_hour_file(
    con: duckdb.DuckDBPyConnection,
    instrument: dict[str, str],
    file_path: str | Path,
    receive_column: str,
    event_column: str,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    receive = ident(receive_column)
    event = ident(event_column)
    symbol = sql_quote(instrument["symbol"])
    quantiles = ", ".join(repr(level) for level in QUANTILE_LEVELS)
    query = f"""
        WITH parsed AS (
            SELECT
                TRY_CAST({receive} AS TIMESTAMPTZ) AS receive_ts,
                TRY_CAST({event} AS TIMESTAMPTZ) AS event_ts
            FROM {scan_expr(file_path)}
            WHERE CAST(symbol AS VARCHAR) = {symbol}
        ),
        selected AS (
            SELECT
                receive_ts,
                event_ts,
                CASE
                    WHEN receive_ts IS NOT NULL
                     AND event_ts IS NOT NULL
                     AND isfinite(receive_ts)
                     AND isfinite(event_ts)
                    THEN datediff('microsecond', event_ts, receive_ts) / 1000.0
                    ELSE NULL
                END AS latency_ms
            FROM parsed
            WHERE receive_ts IS NULL
               OR (
                    receive_ts >= {timestamp_literal(start)}
                AND receive_ts < {timestamp_literal(end)}
               )
        )
        SELECT
            count(*) AS total_rows,
            count(*) FILTER (
                WHERE receive_ts IS NULL OR NOT isfinite(receive_ts)
            ) AS recv_null_or_invalid_rows,
            count(*) FILTER (
                WHERE event_ts IS NULL OR NOT isfinite(event_ts)
            ) AS event_null_or_invalid_rows,
            count(latency_ms) AS valid_latency_rows,
            min(receive_ts) FILTER (WHERE isfinite(receive_ts)) AS first_receive_ts,
            max(receive_ts) FILTER (WHERE isfinite(receive_ts)) AS last_receive_ts,
            min(event_ts) FILTER (WHERE isfinite(event_ts)) AS first_event_ts,
            max(event_ts) FILTER (WHERE isfinite(event_ts)) AS last_event_ts,
            count(*) FILTER (WHERE latency_ms < 0) AS negative_latency_rows,
            count(*) FILTER (WHERE latency_ms = 0) AS zero_latency_rows,
            count(*) FILTER (WHERE latency_ms > 10) AS latency_gt_10ms_rows,
            count(*) FILTER (WHERE latency_ms > 50) AS latency_gt_50ms_rows,
            count(*) FILTER (WHERE latency_ms > 100) AS latency_gt_100ms_rows,
            count(*) FILTER (WHERE latency_ms > 250) AS latency_gt_250ms_rows,
            count(*) FILTER (WHERE latency_ms > 500) AS latency_gt_500ms_rows,
            count(*) FILTER (WHERE latency_ms > 1000) AS latency_gt_1000ms_rows,
            count(*) FILTER (WHERE latency_ms > 5000) AS latency_gt_5000ms_rows,
            min(latency_ms) AS latency_min_ms,
            approx_quantile(latency_ms, [{quantiles}]::FLOAT[]) AS latency_quantiles,
            avg(latency_ms) AS latency_mean_ms,
            stddev_pop(latency_ms) AS latency_stddev_ms,
            max(latency_ms) AS latency_max_ms
        FROM selected
    """
    result = con.execute(query).fetchone()
    if result is None:
        raise RuntimeError(f"No aggregate result for {file_path}")

    (
        total_rows,
        recv_bad,
        event_bad,
        valid_rows,
        first_receive,
        last_receive,
        first_event,
        last_event,
        negative_rows,
        zero_rows,
        gt10,
        gt50,
        gt100,
        gt250,
        gt500,
        gt1000,
        gt5000,
        minimum,
        values,
        mean,
        stddev,
        maximum,
    ) = result
    values = list(values or [None] * len(QUANTILE_LEVELS))
    values += [None] * (len(QUANTILE_LEVELS) - len(values))
    hour_start = hour_from_path(file_path)
    date_text = hour_start.date().isoformat() if hour_start else ""
    hour_text = f"{hour_start.hour:02d}" if hour_start else ""

    valid_rows = int(valid_rows or 0)
    negative_rows = int(negative_rows or 0)
    gt100 = int(gt100 or 0)
    return {
        "exchange": instrument["exchange"],
        "market_type": instrument["market_type"],
        "symbol": instrument["symbol"],
        "base_asset": instrument.get("base_asset", ""),
        "date": date_text,
        "hour": hour_text,
        "file_path": str(file_path),
        "receive_column": receive_column,
        "event_column": event_column,
        "first_receive_ts": first_receive,
        "last_receive_ts": last_receive,
        "first_event_ts": first_event,
        "last_event_ts": last_event,
        "total_rows": int(total_rows or 0),
        "valid_latency_rows": valid_rows,
        "recv_null_or_invalid_rows": int(recv_bad or 0),
        "event_null_or_invalid_rows": int(event_bad or 0),
        "negative_latency_rows": negative_rows,
        "zero_latency_rows": int(zero_rows or 0),
        "latency_gt_10ms_rows": int(gt10 or 0),
        "latency_gt_50ms_rows": int(gt50 or 0),
        "latency_gt_100ms_rows": gt100,
        "latency_gt_250ms_rows": int(gt250 or 0),
        "latency_gt_500ms_rows": int(gt500 or 0),
        "latency_gt_1000ms_rows": int(gt1000 or 0),
        "latency_gt_5000ms_rows": int(gt5000 or 0),
        "negative_latency_ratio": (negative_rows / valid_rows if valid_rows else None),
        "latency_gt_100ms_ratio": gt100 / valid_rows if valid_rows else None,
        "latency_min_ms": minimum,
        "latency_p01_ms": values[0],
        "latency_p50_ms": values[1],
        "latency_p90_ms": values[2],
        "latency_p95_ms": values[3],
        "latency_p99_ms": values[4],
        "latency_p999_ms": values[5],
        "latency_mean_ms": mean,
        "latency_stddev_ms": stddev,
        "latency_max_ms": maximum,
    }


def optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = {
        column: sum(integer(row.get(column)) for row in rows)
        for column in COUNT_COLUMNS
    }
    valid = counts["valid_latency_rows"]
    total = counts["total_rows"]
    weighted_mean_numerator = 0.0
    weighted_mean_denominator = 0
    for row in rows:
        mean = optional_float(row.get("latency_mean_ms"))
        n = integer(row.get("valid_latency_rows"))
        if mean is not None and n > 0:
            weighted_mean_numerator += mean * n
            weighted_mean_denominator += n

    def metric_values(column: str) -> list[float]:
        return [
            value
            for row in rows
            if (value := optional_float(row.get(column))) is not None
        ]

    p01 = metric_values("latency_p01_ms")
    p50 = metric_values("latency_p50_ms")
    p90 = metric_values("latency_p90_ms")
    p95 = metric_values("latency_p95_ms")
    p99 = metric_values("latency_p99_ms")
    p999 = metric_values("latency_p999_ms")
    result: dict[str, object] = {
        "n_hours": len(rows),
        **counts,
        "valid_latency_ratio": valid / total if total else None,
        "negative_latency_ratio": (
            counts["negative_latency_rows"] / valid if valid else None
        ),
        "latency_gt_100ms_ratio": (
            counts["latency_gt_100ms_rows"] / valid if valid else None
        ),
        "event_weighted_mean_ms": (
            weighted_mean_numerator / weighted_mean_denominator
            if weighted_mean_denominator
            else None
        ),
        "median_hour_p01_ms": statistics.median(p01) if p01 else None,
        "median_hour_p50_ms": statistics.median(p50) if p50 else None,
        "p10_hour_p50_ms": linear_quantile(p50, 0.10),
        "p90_hour_p50_ms": linear_quantile(p50, 0.90),
        "median_hour_p90_ms": statistics.median(p90) if p90 else None,
        "median_hour_p95_ms": statistics.median(p95) if p95 else None,
        "median_hour_p99_ms": statistics.median(p99) if p99 else None,
        "median_hour_p999_ms": statistics.median(p999) if p999 else None,
        "min_hour_p50_ms": min(p50) if p50 else None,
        "max_hour_p50_ms": max(p50) if p50 else None,
    }
    return result


def group_summary(
    rows: list[dict[str, str]],
    key_columns: list[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[column] for column in key_columns)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        output.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                **summarize_rows(members),
            }
        )
    return output


def build_summaries(out_dir: Path) -> dict[str, object]:
    hourly_path = out_dir / "receive_event_latency_hourly.csv"
    rows = read_csv_rows(hourly_path)
    if not rows:
        raise RuntimeError(f"No hourly latency rows in {hourly_path}")

    daily_keys = ["exchange", "market_type", "symbol", "date"]
    instrument_keys = ["exchange", "market_type", "symbol"]
    exchange_keys = ["exchange", "market_type"]
    exchange_daily_keys = ["exchange", "market_type", "date"]
    exchange_asset_keys = ["exchange", "market_type", "base_asset"]

    daily = group_summary(rows, daily_keys)
    instruments = group_summary(rows, instrument_keys)
    exchanges = group_summary(rows, exchange_keys)
    exchange_daily = group_summary(rows, exchange_daily_keys)
    exchange_assets = group_summary(rows, exchange_asset_keys)

    atomic_write_rows(
        out_dir / "receive_event_latency_daily.csv",
        daily,
        daily_keys + SUMMARY_METRIC_COLUMNS,
    )
    atomic_write_rows(
        out_dir / "receive_event_latency_instrument.csv",
        instruments,
        instrument_keys + SUMMARY_METRIC_COLUMNS,
    )
    atomic_write_rows(
        out_dir / "receive_event_latency_exchange.csv",
        exchanges,
        exchange_keys + SUMMARY_METRIC_COLUMNS,
    )
    atomic_write_rows(
        out_dir / "receive_event_latency_exchange_daily.csv",
        exchange_daily,
        exchange_daily_keys + SUMMARY_METRIC_COLUMNS,
    )
    atomic_write_rows(
        out_dir / "receive_event_latency_exchange_asset.csv",
        exchange_assets,
        exchange_asset_keys + SUMMARY_METRIC_COLUMNS,
    )

    venue_by_key = {
        (str(row["exchange"]), str(row["market_type"])): row for row in exchanges
    }
    venue_summaries = {
        f"{exchange}/{market_type}": row
        for (exchange, market_type), row in venue_by_key.items()
    }
    metrics = (
        "median_hour_p50_ms",
        "median_hour_p90_ms",
        "median_hour_p95_ms",
        "median_hour_p99_ms",
    )

    def differences(
        left: dict[str, object], right: dict[str, object]
    ) -> dict[str, float | None]:
        return {
            metric: (
                float(left[metric]) - float(right[metric])
                if left.get(metric) is not None and right.get(metric) is not None
                else None
            )
            for metric in metrics
        }

    comparison: dict[str, object] = {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "hourly_rows": len(rows),
        "interpretation": (
            "Positive latency means the local collector received the BBO after "
            "the exchange event timestamp. Venue summaries give every "
            "instrument-hour equal weight through median hourly quantiles. "
            "Every comparison key means left venue minus right venue."
        ),
        "venue_summaries": venue_summaries,
    }

    venue_comparisons = {}
    for label, left_key, right_key in COMPARISON_VENUE_PAIRS:
        left = venue_by_key.get(left_key)
        right = venue_by_key.get(right_key)
        if left is not None and right is not None:
            venue_comparisons[label] = differences(left, right)
    comparison["venue_comparisons_ms"] = venue_comparisons

    by_venue_asset = {
        (row["exchange"], row["market_type"], row["base_asset"]): row
        for row in exchange_assets
        if row.get("base_asset")
    }
    comparisons_by_asset = {}
    for label, left_key, right_key in COMPARISON_VENUE_PAIRS:
        left_assets = {
            base_asset
            for exchange, market_type, base_asset in by_venue_asset
            if (exchange, market_type) == left_key
        }
        right_assets = {
            base_asset
            for exchange, market_type, base_asset in by_venue_asset
            if (exchange, market_type) == right_key
        }
        shared_assets = sorted(left_assets & right_assets)
        if shared_assets:
            comparisons_by_asset[label] = {
                base_asset: differences(
                    by_venue_asset[(*left_key, base_asset)],
                    by_venue_asset[(*right_key, base_asset)],
                )
                for base_asset in shared_assets
            }
    comparison["venue_comparisons_by_base_asset_ms"] = comparisons_by_asset
    config_path = out_dir / "receive_event_latency.CONFIG.json"
    if config_path.is_file():
        comparison["config_hash"] = json.loads(config_path.read_text()).get(
            "config_hash"
        )
    atomic_write_json(out_dir / "receive_event_latency_report.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Low-memory receive_time - event_time latency audit for compacted "
            "Binance perpetual, Binance spot, and Bybit perpetual BBO Parquet "
            "files."
        )
    )
    parser.add_argument("--universe", required=True)
    parser.add_argument("--out", required=True, help="Audit output directory")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--excluded-hours",
        help="Defaults to excluded_hours.csv beside universe.csv when present.",
    )
    parser.add_argument(
        "--event-column",
        default="event_time",
        help="Exchange timestamp column to subtract from local receive time.",
    )
    parser.add_argument(
        "--venues",
        default=",".join(DEFAULT_VENUES),
        help=(
            "Comma-separated exchange/market_type values; default: "
            "binance/perp,binance/spot,bybit/perp."
        ),
    )
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--temp-dir", default="./duckdb_tmp")
    parser.add_argument("--max-temp-size", default="8GB")
    parser.add_argument("--min-free-disk-gb", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reopen-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    if start is None or end is None or start >= end:
        parser.error("--start and --end must define a non-empty UTC range")
    if args.threads <= 0:
        parser.error("--threads must be > 0")
    if args.reopen_every < 0:
        parser.error("--reopen-every must be >= 0")
    if not math.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb must be finite and >= 0")

    try:
        venues = parse_venues(args.venues)
    except ValueError as exc:
        parser.error(str(exc))
    if not venues:
        parser.error("--venues must select at least one venue")

    universe_path = Path(args.universe).expanduser().resolve()
    if not universe_path.is_file():
        raise FileNotFoundError(universe_path)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir = Path(args.temp_dir).expanduser().resolve()
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    excluded_path = resolve_excluded_hours_path(universe_path, args.excluded_hours)
    excluded_by_key = load_excluded_hours(excluded_path)
    universe = [
        row
        for row in read_csv_rows(universe_path)
        if (row.get("exchange"), row.get("market_type")) in venues
    ]
    if not universe:
        raise RuntimeError(
            "No instruments selected from universe.csv for venues " f"{venues}"
        )
    for row in universe:
        if not row.get("parquet_glob"):
            raise RuntimeError(f"Universe row has no parquet_glob: {row}")

    config = {
        "script_version": SCRIPT_VERSION,
        "universe_path": str(universe_path),
        "universe_sha256": sha256_file(universe_path),
        "excluded_hours_path": str(excluded_path) if excluded_path else None,
        "excluded_hours_sha256": sha256_file(excluded_path),
        "start": start.isoformat(timespec="microseconds"),
        "end": end.isoformat(timespec="microseconds"),
        "event_column": args.event_column,
        "venues": [f"{exchange}/{market_type}" for exchange, market_type in venues],
        "latency_definition": "receive_timestamp_minus_event_timestamp_ms",
        "quantile_method": "DuckDB approx_quantile per instrument-hour",
    }
    config["config_hash"] = stable_hash(config)
    config_path = out_dir / "receive_event_latency.CONFIG.json"
    if args.resume:
        if not config_path.is_file():
            raise RuntimeError(
                f"Cannot --resume without configuration manifest {config_path}"
            )
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise RuntimeError(
                "Resume configuration mismatch; use a new --out directory or "
                "restart without --resume"
            )
    else:
        # A failed fresh rerun must not leave an older completed report beside
        # a newly truncated hourly file.
        for name in (
            "receive_event_latency_hourly.csv",
            "receive_event_latency_daily.csv",
            "receive_event_latency_instrument.csv",
            "receive_event_latency_exchange.csv",
            "receive_event_latency_exchange_daily.csv",
            "receive_event_latency_exchange_asset.csv",
            "receive_event_latency_report.json",
        ):
            (out_dir / name).unlink(missing_ok=True)
        atomic_write_json(config_path, config)

    hourly_path = out_dir / "receive_event_latency_hourly.csv"
    completed: set[tuple[str, str, str, str]] = set()
    if args.resume and hourly_path.is_file():
        for row in read_csv_rows(hourly_path):
            completed.add(
                (
                    row["exchange"],
                    row["market_type"],
                    row["symbol"],
                    row["file_path"],
                )
            )
    mode = "a" if args.resume and hourly_path.is_file() else "w"

    con = connect(args)
    processed_since_reopen = 0
    new_rows = 0
    try:
        with hourly_path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HOURLY_COLUMNS)
            if mode == "w":
                writer.writeheader()

            for instrument_index, instrument in enumerate(universe, 1):
                key = (
                    instrument["exchange"],
                    instrument["market_type"],
                    instrument["symbol"],
                )
                excluded = excluded_by_key.get(key, set())
                files = []
                for file_path in glob.iglob(instrument["parquet_glob"]):
                    if not file_overlaps_range(file_path, start, end):
                        continue
                    if hour_from_path(file_path) in excluded:
                        continue
                    files.append(file_path)
                files.sort(
                    key=lambda value: (hour_from_path(value) or datetime.min, value)
                )
                if not files:
                    print(
                        f"[WARN] {instrument_index}/{len(universe)} no files for "
                        f"{'/'.join(key)}"
                    )
                    continue

                schema = columns_by_name(con, files[0])
                receive_column = choose_receive_column(set(schema))
                if args.event_column not in schema:
                    raise RuntimeError(
                        f"{args.event_column!r} missing for {'/'.join(key)}; "
                        f"available={sorted(schema)}"
                    )
                ensure_timestamp_like(receive_column, schema[receive_column], files[0])
                ensure_timestamp_like(
                    args.event_column, schema[args.event_column], files[0]
                )
                print(
                    f"[INFO] {instrument_index}/{len(universe)} {'/'.join(key)} "
                    f"files={len(files)} receive={receive_column} "
                    f"event={args.event_column}"
                )

                for file_index, file_path in enumerate(files, 1):
                    resume_key = (*key, file_path)
                    if resume_key in completed:
                        continue
                    if (
                        file_index == 1
                        or file_index % 50 == 0
                        or file_index == len(files)
                    ):
                        print(f"[INFO]   file {file_index}/{len(files)}")

                    for disk_path, label in (
                        (out_dir, "output"),
                        (args.temp_dir, "temp"),
                    ):
                        free_gb = shutil.disk_usage(disk_path).free / 1024**3
                        if free_gb < args.min_free_disk_gb:
                            raise RuntimeError(
                                f"{label} volume has only {free_gb:.2f} GiB free"
                            )

                    result = scan_hour_file(
                        con,
                        instrument,
                        file_path,
                        receive_column,
                        args.event_column,
                        start,
                        end,
                    )
                    writer.writerow(
                        {column: result.get(column) for column in HOURLY_COLUMNS}
                    )
                    handle.flush()
                    completed.add(resume_key)
                    new_rows += 1
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

    comparison = build_summaries(out_dir)
    deltas = comparison.get("venue_comparisons_ms")
    print(f"[DONE] processed new hourly files: {new_rows}")
    print(f"[DONE] wrote hourly audit: {hourly_path}")
    print(f"[DONE] wrote summaries under: {out_dir}")
    if deltas:
        print(
            "[RESULT] Venue median-hour latency-quantile differences (ms): "
            + json.dumps(deltas, sort_keys=True)
        )


if __name__ == "__main__":
    main()
