#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from datetime import datetime, time, timedelta
from pathlib import Path

import duckdb

from common import (
    build_scan_expr,
    build_ts_expr,
    choose_timestamp_col,
    get_columns,
    load_excluded_hours,
    parse_dt,
    read_csv_rows,
    resolve_excluded_hours_path,
    sql_quote,
    ts_lit,
)


DATE_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})")
CACHE_CONFIG_VERSION = 2


def quality_by_key(path: str) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (r["exchange"], r["market_type"], r["symbol"]): r
        for r in read_csv_rows(path)
    }


def is_bad_number(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


def parse_positive_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if is_bad_number(x) or x <= 0:
        return None
    return x


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def next_midnight(dt: datetime) -> datetime:
    return datetime.combine(dt.date() + timedelta(days=1), time())


def hour_from_path(path: str) -> datetime | None:
    date_value = None
    hour_value = None
    for part in Path(path).parts:
        if part.startswith("date="):
            date_value = part.split("=", 1)[1]
        elif part.startswith("hour="):
            hour_value = part.split("=", 1)[1]
    if date_value is None or hour_value is None:
        return None
    try:
        return datetime.fromisoformat(
            f"{date_value} {int(hour_value):02d}:00:00"
        )
    except (TypeError, ValueError):
        return None


def iter_chunks(start: datetime, end: datetime, chunk_minutes: float):
    if chunk_minutes <= 0:
        raise ValueError("--chunk-minutes must be > 0")

    step = timedelta(minutes=chunk_minutes)
    cur = start
    while cur < end:
        # Do not let one output file cross midnight; this keeps date=YYYY-MM-DD correct.
        nxt = min(cur + step, next_midnight(cur), end)
        if nxt <= cur:
            raise RuntimeError(f"bad chunk range: {cur} -> {nxt}")
        yield cur, nxt
        cur = nxt


def replace_partition_if_wildcard(pattern: str, key: str, value: str) -> str:
    """
    Replace path partition key=* only when that partition is actually wildcarded.

    Examples:
      /x/date=*/hour=*/*.parquet -> /x/date=2026-06-26/hour=13/*.parquet
      /x/date=2026-06-26/hour=*/*.parquet -> date is kept fixed, hour is replaced
    """
    m = re.search(rf"({re.escape(key)}=)([^/]+)", pattern)
    if not m:
        return pattern

    current_value = m.group(2)
    if not any(ch in current_value for ch in "*?["):
        return pattern

    return pattern[: m.start(2)] + value + pattern[m.end(2) :]


def hourly_candidate_globs(pattern: str, scan_start: datetime, scan_end: datetime) -> list[str]:
    """Return a small list of candidate globs for the time range.

    This is the main low-RAM / low-I/O trick: if the parquet path contains
    date=*/hour=* partitions, each chunk only scans the needed hour folders
    instead of repeatedly scanning the whole universe glob.
    """
    if scan_end <= scan_start:
        return []

    # If the query ends exactly on an hour boundary, rows at that boundary are excluded.
    last_ts = scan_end - timedelta(microseconds=1)
    cur = floor_to_hour(scan_start)
    last_hour = floor_to_hour(last_ts)

    out: list[str] = []
    seen: set[str] = set()
    while cur <= last_hour:
        p = replace_partition_if_wildcard(pattern, "date", cur.date().isoformat())
        p = replace_partition_if_wildcard(p, "hour", f"{cur.hour:02d}")
        if p not in seen:
            seen.add(p)
            out.append(p)
        cur += timedelta(hours=1)

    return out or [pattern]


def list_files_for_range(
    pattern: str,
    scan_start: datetime,
    scan_end: datetime,
    excluded_hours: set[datetime] | None = None,
) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    excluded = excluded_hours if excluded_hours is not None else set()

    for candidate in hourly_candidate_globs(pattern, scan_start, scan_end):
        for f in glob.iglob(candidate, recursive=True):
            if (
                f not in seen
                and Path(f).is_file()
                and hour_from_path(f) not in excluded
            ):
                seen.add(f)
                files.append(f)

    files.sort()
    return files


def first_file_for_pattern(
    pattern: str,
    excluded_hours: set[datetime] | None = None,
) -> str | None:
    excluded = excluded_hours if excluded_hours is not None else set()
    for f in glob.iglob(pattern, recursive=True):
        if (
            Path(f).is_file()
            and hour_from_path(f) not in excluded
        ):
            return f
    return None


def first_file_for_range(
    pattern: str,
    scan_start: datetime,
    scan_end: datetime,
    excluded_hours: set[datetime] | None = None,
) -> str | None:
    """Find one schema file without storing every file path in the full range."""
    excluded = excluded_hours if excluded_hours is not None else set()
    for candidate in hourly_candidate_globs(pattern, scan_start, scan_end):
        best: str | None = None
        for f in glob.iglob(candidate, recursive=True):
            if (
                Path(f).is_file()
                and hour_from_path(f) not in excluded
                and (best is None or f < best)
            ):
                best = f
        if best is not None:
            return best
    return None


def scan_expr_for_files(files: list[str]) -> str:
    if not files:
        raise ValueError("scan_expr_for_files got empty file list")

    if len(files) == 1:
        file_arg = sql_quote(files[0])
    else:
        file_arg = "[" + ", ".join(sql_quote(f) for f in files) + "]"

    return f"read_parquet({file_arg}, union_by_name=true, hive_partitioning=true)"


def infer_date_bounds_from_paths(
    pattern: str,
    excluded_hours: set[datetime] | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Infer [start, end) from date=YYYY-MM-DD path partitions without reading parquet rows."""
    lo: datetime | None = None
    hi: datetime | None = None
    excluded = excluded_hours if excluded_hours is not None else set()
    for f in glob.iglob(pattern, recursive=True):
        if hour_from_path(f) in excluded:
            continue
        m = DATE_RE.search(f)
        if m:
            dt = datetime.fromisoformat(m.group(1))
            lo = dt if lo is None else min(lo, dt)
            hi = dt if hi is None else max(hi, dt)

    if lo is None or hi is None:
        return None, None

    return lo, hi + timedelta(days=1)


def ensure_cache_config(target: Path, config: dict) -> None:
    """Prevent cache files produced by incompatible runs from being mixed."""
    manifest = target / "_cache_config.json"
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read cache config {manifest}: {exc}") from exc
        if existing != config:
            raise RuntimeError(
                f"cache config mismatch under {target}; use a new --out directory "
                "or remove this instrument cache before rebuilding"
            )
        return

    if next(target.rglob("*.parquet"), None) is not None:
        raise RuntimeError(
            f"existing parquet cache has no config manifest under {target}; "
            "use a new --out directory or remove this instrument cache"
        )

    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest)


def infer_bounds_by_scanning_duckdb(con: duckdb.DuckDBPyConnection, scan: str, ts_expr: str) -> tuple[datetime | None, datetime | None]:
    """Fallback when paths do not contain date=YYYY-MM-DD partitions."""
    mn, mx = con.execute(f"SELECT min({ts_expr}), max({ts_expr}) FROM {scan}").fetchone()
    if mn is None or mx is None:
        return None, None
    return datetime.combine(mn.date(), time()), datetime.combine(mx.date() + timedelta(days=1), time())


def resolve_bounds(
    con: duckdb.DuckDBPyConnection,
    pattern: str,
    full_scan: str,
    ts_expr: str,
    requested_start: datetime | None,
    requested_end: datetime | None,
    excluded_hours: set[datetime] | None = None,
) -> tuple[datetime | None, datetime | None]:
    if requested_start is not None and requested_end is not None:
        return requested_start, requested_end

    path_start, path_end = infer_date_bounds_from_paths(pattern, excluded_hours)
    if path_start is None or path_end is None:
        path_start, path_end = infer_bounds_by_scanning_duckdb(con, full_scan, ts_expr)

    if path_start is None or path_end is None:
        return None, None

    start = requested_start or path_start
    end = requested_end or path_end
    return start, end


def resolve_bounds_from_paths(
    pattern: str,
    requested_start: datetime | None,
    requested_end: datetime | None,
    excluded_hours: set[datetime] | None = None,
) -> tuple[datetime | None, datetime | None]:
    if requested_start is not None and requested_end is not None:
        return requested_start, requested_end

    path_start, path_end = infer_date_bounds_from_paths(pattern, excluded_hours)
    if path_start is None or path_end is None:
        return None, None

    return requested_start or path_start, requested_end or path_end


def build_copy_sql(
    *,
    row: dict[str, str],
    scan: str,
    ts_expr: str,
    price_mode: str,
    max_interval_ms: float,
    max_spread_bps: float,
    min_interval_ms: float,
    scan_start: datetime,
    chunk_start: datetime,
    chunk_end: datetime,
    excluded_hours: set[datetime],
    drop_zero_returns: bool,
    out_file: Path,
) -> str:
    zero_filter = "AND ABS(ret) > 1e-14" if drop_zero_returns else ""
    selected_price = "mid" if price_mode == "mid" else "microprice"
    excluded_interval_filters = "".join(
        "\n              AND NOT ("
        f"start_ts < {ts_lit(hour_start + timedelta(hours=1))} "
        f"AND end_ts > {ts_lit(hour_start)})"
        for hour_start in sorted(excluded_hours)
    )

    # Compacted input is assumed to be strictly increasing and unique by
    # receive timestamp. Empty OVER clauses follow preserved input order and
    # avoid ROW_NUMBER, deduplication, and timestamp sorting.

    scan_time_filter = f"ts >= {ts_lit(scan_start)} AND ts < {ts_lit(chunk_end)}"
    output_time_filter = f"end_ts >= {ts_lit(chunk_start)} AND end_ts < {ts_lit(chunk_end)}"

    return f"""
    COPY (
        WITH raw AS (
            SELECT
                {sql_quote(row['exchange'])} AS exchange,
                {sql_quote(row['market_type'])} AS market_type,
                CAST(symbol AS VARCHAR) AS symbol,
                {ts_expr} AS ts,
                CAST(bid_price AS DOUBLE) AS bid_price,
                CAST(ask_price AS DOUBLE) AS ask_price,
                CAST(bid_qty AS DOUBLE) AS bid_qty,
                CAST(ask_qty AS DOUBLE) AS ask_qty
            FROM {scan}
            WHERE CAST(symbol AS VARCHAR) = {sql_quote(row['symbol'])}
        ),
        enriched AS (
            SELECT
                *,
                (bid_price + ask_price) / 2.0 AS mid,
                CASE
                    WHEN bid_qty > 0 AND ask_qty > 0
                    THEN (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
                    ELSE NULL
                END AS microprice,
                10000.0 * (ask_price - bid_price) / ((bid_price + ask_price) / 2.0) AS spread_bps
            FROM raw
            WHERE {scan_time_filter}
              AND ts IS NOT NULL
              AND isfinite(ts)
              AND bid_price IS NOT NULL
              AND ask_price IS NOT NULL
              AND bid_qty IS NOT NULL
              AND ask_qty IS NOT NULL
              AND isfinite(bid_price)
              AND isfinite(ask_price)
              AND isfinite(bid_qty)
              AND isfinite(ask_qty)
              AND bid_price > 0
              AND ask_price > 0
              AND bid_qty > 0
              AND ask_qty > 0
              AND bid_price < ask_price
        ),
        filtered AS (
            SELECT *
            FROM enriched
            WHERE spread_bps IS NOT NULL
              AND isfinite(spread_bps)
              AND spread_bps BETWEEN 0 AND {max_spread_bps}
              AND {selected_price} IS NOT NULL
              AND isfinite({selected_price})
              AND {selected_price} > 0
        ),
        prices AS (
            SELECT
                exchange,
                market_type,
                symbol,
                ts,
                spread_bps,
                {selected_price} AS price
            FROM filtered
        ),
        lagged AS (
            SELECT
                exchange,
                market_type,
                symbol,
                LAG(ts) OVER () AS start_ts,
                ts AS end_ts,
                LAG(price) OVER () AS prev_price,
                price,
                spread_bps,
                CAST(ts AS DATE) AS date
            FROM prices
        ),
        intervals AS (
            SELECT
                exchange,
                market_type,
                symbol,
                epoch_ns(start_ts) AS start_ns,
                epoch_ns(end_ts) AS end_ns,
                datediff('microsecond', start_ts, end_ts) / 1000.0 AS interval_ms,
                ln(price) - ln(prev_price) AS ret,
                spread_bps,
                date
            FROM lagged
            WHERE start_ts IS NOT NULL
              AND prev_price IS NOT NULL
              AND prev_price > 0
              AND price > 0
              {excluded_interval_filters}
              AND {output_time_filter}
        )
        SELECT
            start_ns,
            end_ns,
            interval_ms,
            ret,
            spread_bps,
            {max_interval_ms} AS max_interval_ms
        FROM intervals
        WHERE isfinite(interval_ms)
          AND isfinite(ret)
          AND isfinite(spread_bps)
          AND interval_ms > {min_interval_ms}
          AND interval_ms <= {max_interval_ms}
          {zero_filter}
    )
    TO {sql_quote(str(out_file))}
    (FORMAT PARQUET, COMPRESSION ZSTD, PRESERVE_ORDER true);
    """


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean return interval parquet cache per instrument, using small time chunks to reduce RAM."
    )
    parser.add_argument("--universe", required=True)
    parser.add_argument(
        "--excluded-hours",
        help=(
            "Optional abnormal-hour CSV. By default excluded_hours.csv beside "
            "the universe is used when present."
        ),
    )
    parser.add_argument("--quality", required=True)
    parser.add_argument("--out", required=True, help="interval_cache directory")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--timestamp-col")
    parser.add_argument("--ts-unit", choices=["timestamp", "ns", "us", "ms"], default="timestamp")
    parser.add_argument("--price-mode", choices=["mid", "microprice"], default="mid")
    parser.add_argument("--min-interval-ms", type=float, default=0.0)
    parser.add_argument("--drop-zero-returns", action="store_true", default=True)
    parser.add_argument("--keep-zero-returns", dest="drop_zero_returns", action="store_false")
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Recompute and atomically replace chunk files that already exist.",
    )
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--temp-dir", default="./duckdb_tmp")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=60.0,
        help="Process each instrument in time chunks. Lower this to 15 or 10 if RAM is still tight.",
    )
    parser.add_argument(
        "--boundary-extra-ms",
        type=float,
        default=1000.0,
        help="Extra lookback before each chunk. Effective lookback = max_interval_ms + boundary_extra_ms.",
    )
    parser.add_argument(
        "--price-mode-in-path",
        action="store_true",
        default=True,
        help="Write under price_mode=mid/microprice to avoid mixing runs.",
    )
    parser.add_argument(
        "--no-price-mode-in-path",
        dest="price_mode_in_path",
        action="store_false",
        help="Use old path layout without price_mode=... in the directory.",
    )
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if args.chunk_minutes <= 0:
        raise SystemExit("--chunk-minutes must be > 0")
    if args.min_interval_ms < 0:
        raise SystemExit("--min-interval-ms must be >= 0")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    requested_start = parse_dt(args.start)
    requested_end = parse_dt(args.end)
    if requested_start and requested_end and requested_start >= requested_end:
        raise SystemExit("--start must be earlier than --end")

    excluded_path = resolve_excluded_hours_path(
        args.universe, args.excluded_hours
    )
    excluded_by_key = load_excluded_hours(excluded_path)
    excluded_count = sum(len(hours) for hours in excluded_by_key.values())
    if excluded_path is not None:
        print(
            f"[INFO] excluded abnormal hours={excluded_count} "
            f"manifest={excluded_path}"
        )

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC';")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)};")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir)};")
    con.execute(f"SET threads = {args.threads};")
    con.execute("SET preserve_insertion_order = true;")
    print(
        "[ASSUMPTION] input receive timestamps are strictly increasing "
        "and unique in parquet file order"
    )

    q = quality_by_key(args.quality)
    rows = read_csv_rows(args.universe)

    for i, row in enumerate(rows, 1):
        key = (row["exchange"], row["market_type"], row["symbol"])
        excluded_hours = excluded_by_key.get(key, set())
        quality = q.get(key)
        if quality is None:
            print(f"[WARN] skip {key}: no quality/cap row")
            continue

        max_interval_ms = parse_positive_float(quality.get("recommended_max_interval_ms"))
        max_spread_bps = parse_positive_float(quality.get("recommended_max_spread_bps"))
        if max_interval_ms is None or max_spread_bps is None:
            print(
                f"[WARN] skip {key}: bad caps "
                f"interval={quality.get('recommended_max_interval_ms')} "
                f"spread={quality.get('recommended_max_spread_bps')}"
            )
            continue

        start, end = resolve_bounds_from_paths(
            row["parquet_glob"],
            requested_start,
            requested_end,
            excluded_hours,
        )
        if start is None or end is None:
            first_file = first_file_for_pattern(
                row["parquet_glob"], excluded_hours
            )
            if first_file is None:
                print(f"[WARN] skip {key}: no parquet files matched {row['parquet_glob']}")
                continue

            first_scan = build_scan_expr(first_file, True)
            cols = get_columns(con, first_scan)
            ts_col = choose_timestamp_col(cols, args.timestamp_col or quality.get("timestamp_col") or None)
            ts_expr = build_ts_expr(ts_col, args.ts_unit)
            full_scan = build_scan_expr(row["parquet_glob"], True)
            start, end = resolve_bounds(
                con,
                row["parquet_glob"],
                full_scan,
                ts_expr,
                requested_start,
                requested_end,
                excluded_hours,
            )
        if start is None or end is None or start >= end:
            print(f"[WARN] skip {key}: could not resolve non-empty time range")
            continue

        schema_file = first_file_for_range(
            row["parquet_glob"], start, end, excluded_hours
        )
        if schema_file is None:
            print(f"[WARN] skip {key}: no parquet files in range [{start}, {end})")
            continue

        schema_scan = build_scan_expr(schema_file, True)
        cols = get_columns(con, schema_scan)
        ts_col = choose_timestamp_col(cols, args.timestamp_col or quality.get("timestamp_col") or None)
        ts_expr = build_ts_expr(ts_col, args.ts_unit)
        target = (
            out_dir
            / f"exchange={row['exchange']}"
            / f"market_type={row['market_type']}"
            / f"symbol={row['symbol']}"
        )
        if args.price_mode_in_path:
            target = target / f"price_mode={args.price_mode}"
        target.mkdir(parents=True, exist_ok=True)

        cache_config = {
            "cache_config_version": CACHE_CONFIG_VERSION,
            "exchange": row["exchange"],
            "market_type": row["market_type"],
            "symbol": row["symbol"],
            "source_parquet_glob": row["parquet_glob"],
            "start": start.isoformat(timespec="microseconds"),
            "end": end.isoformat(timespec="microseconds"),
            "timestamp_col": ts_col,
            "ts_unit": args.ts_unit,
            "price_mode": args.price_mode,
            "input_mode": "streaming_sorted_unique",
            "chunk_minutes": args.chunk_minutes,
            "boundary_extra_ms": args.boundary_extra_ms,
            "min_interval_ms": args.min_interval_ms,
            "max_interval_ms": max_interval_ms,
            "max_spread_bps": max_spread_bps,
            "drop_zero_returns": args.drop_zero_returns,
            "zero_return_epsilon": 1e-14 if args.drop_zero_returns else None,
            "requires_positive_finite_prices_and_quantities": True,
            "requires_bid_strictly_below_ask": True,
        }
        run_excluded_hours = sorted(
            hour_start
            for hour_start in excluded_hours
            if start <= hour_start < end
        )
        if run_excluded_hours:
            cache_config["excluded_hours_utc"] = [
                hour_start.isoformat(timespec="seconds")
                for hour_start in run_excluded_hours
            ]
        ensure_cache_config(target, cache_config)

        print(
            f"[INFO] {i}/{len(rows)} {row['exchange']}/{row['market_type']}/{row['symbol']} "
            f"range=[{start}, {end}) chunk={args.chunk_minutes:g}min "
            f"cap={max_interval_ms:.3f}ms spread={max_spread_bps:.3f}bps "
            f"mem={args.memory_limit} threads={args.threads} "
            f"excluded_hours={len(run_excluded_hours)} "
            "mode=streaming-sorted-input"
        )

        lookback_ms = max_interval_ms + max(0.0, args.boundary_extra_ms)
        written = 0
        skipped_no_files = 0
        skipped_existing = 0

        for chunk_start, chunk_end in iter_chunks(start, end, args.chunk_minutes):
            scan_start = chunk_start - timedelta(milliseconds=lookback_ms)
            files = list_files_for_range(
                row["parquet_glob"],
                scan_start,
                chunk_end,
                excluded_hours,
            )
            if not files:
                skipped_no_files += 1
                continue

            scan = scan_expr_for_files(files)
            date_dir = target / f"date={chunk_start.date().isoformat()}"
            date_dir.mkdir(parents=True, exist_ok=True)
            out_file = date_dir / (
                f"{args.price_mode}_"
                f"chunk_start={chunk_start:%Y%m%dT%H%M%S}_"
                f"chunk_end={chunk_end:%Y%m%dT%H%M%S}.parquet"
            )
            if out_file.exists() and not args.rebuild_existing:
                skipped_existing += 1
                continue

            tmp_file = out_file.with_name(out_file.name + ".tmp")
            if tmp_file.exists():
                tmp_file.unlink()

            sql = build_copy_sql(
                row=row,
                scan=scan,
                ts_expr=ts_expr,
                price_mode=args.price_mode,
                max_interval_ms=max_interval_ms,
                max_spread_bps=max_spread_bps,
                min_interval_ms=args.min_interval_ms,
                scan_start=scan_start,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                excluded_hours={
                    hour_start
                    for hour_start in run_excluded_hours
                    if hour_start < chunk_end
                    and hour_start + timedelta(hours=1) > scan_start
                },
                drop_zero_returns=args.drop_zero_returns,
                out_file=tmp_file,
            )
            try:
                con.execute(sql)
                tmp_file.replace(out_file)
            except Exception:
                if tmp_file.exists():
                    tmp_file.unlink()
                raise
            written += 1

        print(
            f"[INFO] done {row['exchange']}/{row['market_type']}/{row['symbol']}: "
            f"wrote_chunks={written} skipped_existing={skipped_existing} "
            f"skipped_no_files={skipped_no_files} target={target}"
        )

    print(f"[DONE] wrote interval cache under {out_dir}")


if __name__ == "__main__":
    main()
