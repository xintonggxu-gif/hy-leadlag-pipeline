#!/usr/bin/env python3
"""Aggregate the small per-file CSV produced by 02_scan_quality_low_memory.py."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import duckdb


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate partial quality statistics (stage 2 of 2)."
    )
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-cap-ms", type=float, default=50.0)
    parser.add_argument("--hard-cap-ms", type=float, default=2000.0)
    parser.add_argument("--min-spread-bps", type=float, default=10.0)
    parser.add_argument("--hard-spread-bps", type=float, default=100.0)
    parser.add_argument("--memory-limit", default="512MB")
    args = parser.parse_args()

    for name in (
        "min_cap_ms",
        "hard_cap_ms",
        "min_spread_bps",
        "hard_spread_bps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.min_cap_ms > args.hard_cap_ms:
        parser.error("--min-cap-ms must be <= --hard-cap-ms")
    if args.min_spread_bps > args.hard_spread_bps:
        parser.error("--min-spread-bps must be <= --hard-spread-bps")

    partial = Path(args.partial).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not partial.is_file():
        raise FileNotFoundError(partial)
    if partial == out:
        parser.error("--partial and --out must be different files")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    temp_out.unlink(missing_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
        con.execute("SET threads = 1")
        query = f"""
            WITH hourly AS (
                SELECT *
                FROM read_csv_auto(
                    {sql_quote(str(partial))},
                    header = true,
                    sample_size = -1
                )
            ),
            quality AS (
                SELECT
                    exchange,
                    market_type,
                    symbol,
                    ANY_VALUE(parquet_glob) AS parquet_glob,
                    ANY_VALUE(timestamp_col) AS timestamp_col,
                    COUNT(*) AS n_files_scanned,
                    SUM(total_rows) AS total_rows,
                    SUM(clean_rows) AS clean_rows,
                    MIN(first_ts) AS first_ts,
                    MAX(last_ts) AS last_ts,
                    SUM(duplicate_ts) AS duplicate_ts,
                    SUM(backward_ts) AS backward_ts,
                    SUM(crossed_rows) AS crossed_rows,
                    SUM(locked_rows) AS locked_rows,
                    SUM(non_positive_bid) AS non_positive_bid,
                    SUM(non_positive_ask) AS non_positive_ask,
                    SUM(non_positive_bid_qty) AS non_positive_bid_qty,
                    SUM(non_positive_ask_qty) AS non_positive_ask_qty,
                    MEDIAN(gap_p50_ms) AS gap_p50_ms,
                    MEDIAN(gap_p90_ms) AS gap_p90_ms,
                    MEDIAN(gap_p95_ms) AS gap_p95_ms,
                    MEDIAN(gap_p99_ms) AS gap_p99_ms,
                    MEDIAN(gap_p999_ms) AS gap_p999_ms,
                    MAX(gap_max_ms) AS gap_max_ms,
                    SUM(zero_ret_count)::DOUBLE
                        / NULLIF(SUM(ret_count), 0) AS zero_ret_ratio,
                    MEDIAN(spread_bps_p50) AS spread_bps_p50,
                    MEDIAN(spread_bps_p99) AS spread_bps_p99,
                    COUNT(*) FILTER (
                        WHERE processing_mode = 'sorted_fallback'
                    ) AS n_sorted_fallback_files
                FROM hourly
                WHERE total_rows > 0
                GROUP BY exchange, market_type, symbol
            )
            SELECT
                exchange,
                market_type,
                symbol,
                parquet_glob,
                timestamp_col,
                n_files_scanned,
                total_rows,
                clean_rows,
                first_ts,
                last_ts,
                duplicate_ts,
                backward_ts,
                crossed_rows,
                locked_rows,
                non_positive_bid,
                non_positive_ask,
                non_positive_bid_qty,
                non_positive_ask_qty,
                gap_p50_ms,
                gap_p90_ms,
                gap_p95_ms,
                gap_p99_ms,
                gap_p999_ms,
                gap_max_ms,
                zero_ret_ratio,
                spread_bps_p50,
                spread_bps_p99,
                n_sorted_fallback_files,
                GREATEST(
                    {args.min_cap_ms!r},
                    LEAST(
                        {args.hard_cap_ms!r},
                        3.0 * COALESCE(
                            gap_p90_ms, {args.hard_cap_ms!r}
                        ),
                        COALESCE(gap_p99_ms, {args.hard_cap_ms!r})
                    )
                ) AS recommended_max_interval_ms,
                LEAST(
                    {args.hard_spread_bps!r},
                    GREATEST(
                        {args.min_spread_bps!r},
                        3.0 * COALESCE(
                            spread_bps_p99, {args.min_spread_bps!r}
                        )
                    )
                ) AS recommended_max_spread_bps,
                CASE
                    WHEN clean_rows < 100 THEN 'too_few_clean_rows'
                    WHEN backward_ts > 0 THEN 'input_order_problem'
                    WHEN gap_p90_ms >= 500
                        THEN 'slow_update: stale filtering important'
                    WHEN gap_p99_ms > 5.0 * NULLIF(gap_p90_ms, 0)
                        THEN 'bursty_or_stale: cap important'
                    WHEN zero_ret_ratio >= 0.90
                        THEN 'many_zero_returns: check symbol/liquidity'
                    ELSE 'normal'
                END AS suggested_action
            FROM quality
            ORDER BY exchange, market_type, symbol
        """
        con.execute(f"""
            COPY ({query})
            TO {sql_quote(str(temp_out))}
            (HEADER, DELIMITER ',')
        """)
        os.replace(temp_out, out)
    finally:
        con.close()
        temp_out.unlink(missing_ok=True)

    print(f"[DONE] wrote instrument quality: {out}")


if __name__ == "__main__":
    main()
