#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path

import duckdb

from common import (
    build_scan_expr,
    choose_timestamp_basis_col,
    get_columns,
    write_csv_rows,
    write_json,
)

TARGET_INSTRUMENTS: dict[tuple[str, str, str], str] = {
    # Binance spot (group B).
    ("binance", "spot", "XRPUSDT"): "XRP",
    ("binance", "spot", "DOGEUSDT"): "DOGE",
    ("binance", "spot", "ETHUSDT"): "ETH",
    ("binance", "spot", "SOLUSDT"): "SOL",
    ("binance", "spot", "PEPEUSDT"): "1000PEPE",
    ("binance", "spot", "XRPUSDC"): "XRP",
    ("binance", "spot", "DOGEUSDC"): "DOGE",
    ("binance", "spot", "ETHUSDC"): "ETH",
    ("binance", "spot", "SOLUSDC"): "SOL",
    ("binance", "spot", "PEPEUSDC"): "1000PEPE",
    ("binance", "spot", "USDCUSDT"): "USDC",

    # Binance perpetuals (group A).
    ("binance", "perp", "XRPUSDT"): "XRP",
    ("binance", "perp", "DOGEUSDT"): "DOGE",
    ("binance", "perp", "FARTCOINUSDT"): "FARTCOIN",
    ("binance", "perp", "ETHUSDT"): "ETH",
    ("binance", "perp", "SOLUSDT"): "SOL",
    ("binance", "perp", "1000PEPEUSDT"): "1000PEPE",
    ("binance", "perp", "XRPUSDC"): "XRP",
    ("binance", "perp", "DOGEUSDC"): "DOGE",
    ("binance", "perp", "ETHUSDC"): "ETH",
    ("binance", "perp", "SOLUSDC"): "SOL",
    ("binance", "perp", "1000PEPEUSDC"): "1000PEPE",
    ("binance", "perp", "USDCUSDT"): "USDC",

    # Bybit linear contracts are stored under market_type=perp (group B).
    # The generic PERP symbols below are real source symbols, not aliases for the
    # USDT contracts, so retain both variants as distinct instruments.
    ("bybit", "perp", "XRPUSDT"): "XRP",
    ("bybit", "perp", "XRPPERP"): "XRP",
    ("bybit", "perp", "DOGEUSDT"): "DOGE",
    ("bybit", "perp", "DOGEPERP"): "DOGE",
    ("bybit", "perp", "FARTCOINUSDT"): "FARTCOIN",
    ("bybit", "perp", "ETHUSDT"): "ETH",
    ("bybit", "perp", "ETHPERP"): "ETH",
    ("bybit", "perp", "SOLUSDT"): "SOL",
    ("bybit", "perp", "SOLPERP"): "SOL",
    ("bybit", "perp", "1000PEPEUSDT"): "1000PEPE",
    ("bybit", "perp", "1000PEPEPERP"): "1000PEPE",
    ("bybit", "perp", "USDCUSDT"): "USDC",
}


def parse_compacted_path(root: Path, path: Path) -> dict[str, str] | None:
    """
    Parse a compacted parquet path of the form:

    exchange/
    market_type/
    feed/
    symbol=SYMBOL/
    date=YYYY-MM-DD/
    hour=HH/
    *.parquet
    """
    rel = path.relative_to(root)
    parts = rel.parts

    if len(parts) < 7 or path.suffix != ".parquet":
        return None

    exchange, market_type, feed = parts[0], parts[1], parts[2]
    symbol_part, date_part, hour_part = parts[3], parts[4], parts[5]

    if not symbol_part.startswith("symbol="):
        return None
    if not date_part.startswith("date="):
        return None
    if not hour_part.startswith("hour="):
        return None

    return {
        "exchange": exchange,
        "market_type": market_type,
        "feed": feed,
        "symbol": symbol_part.split("=", 1)[1],
        "date": date_part.split("=", 1)[1],
        "hour": hour_part.split("=", 1)[1],
    }


def assign_group(exchange: str, market_type: str) -> str:
    """
    Grouping rule:

    A:
        Binance perpetual instruments.

    B:
        All other instruments.
    """
    if exchange.lower() == "binance" and market_type.lower() == "perp":
        return "A"

    return "B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build instrument universe from compacted hourly parquet tree.")
    parser.add_argument("--root", required=True, help="Example: /home/xintong/market_data_compacted")
    parser.add_argument("--out", required=True, help="Output directory for universe.csv and task_config.json")
    parser.add_argument("--feed", default="bbo")
    parser.add_argument("--exchange", action="append", help="Optional exchange filter; repeatable.")
    parser.add_argument("--market-type", action="append", help="Optional market-type filter; repeatable.")
    parser.add_argument("--symbol", action="append", help="Optional symbol filter; repeatable.")
    parser.add_argument("--date", action="append", help="Optional YYYY-MM-DD partition filter; repeatable.")
    parser.add_argument(
        "--timestamp-basis",
        choices=["receive", "event", "transaction"],
        default="receive",
        help=(
            "Keep only instruments whose compacted schema supports this HY "
            "clock. Binance Spot is excluded from transaction runs."
        ),
    )
    args = parser.parse_args()

    selected_exchanges = {value.lower() for value in args.exchange or []}
    selected_market_types = {value.lower() for value in args.market_type or []}
    selected_symbols = {value.upper() for value in args.symbol or []}
    selected_dates = set(args.date or [])

    root = Path(args.root)
    out_dir = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Compacted parquet root does not exist: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str, str], dict] = {}
    ignored_instruments: dict[tuple[str, str, str], str] = {}
    hour_file_counts: dict[
        tuple[str, str, str, str, str, str], int
    ] = defaultdict(int)

    exchange_patterns = sorted(selected_exchanges) or ["*"]
    market_type_patterns = sorted(selected_market_types) or ["*"]
    symbol_patterns = sorted(selected_symbols) or ["*"]
    date_patterns = sorted(selected_dates) or ["*"]
    parquet_patterns = [
        f"{exchange}/{market_type}/{args.feed}/symbol={symbol}/date={date}/hour=*/*.parquet"
        for exchange in exchange_patterns
        for market_type in market_type_patterns
        for symbol in symbol_patterns
        for date in date_patterns
    ]

    for path in chain.from_iterable(root.glob(pattern) for pattern in parquet_patterns):
        rec = parse_compacted_path(root, path)
        if rec is None or rec["feed"] != args.feed:
            continue

        target_key = (
            rec["exchange"].lower(),
            rec["market_type"].lower(),
            rec["symbol"].upper(),
        )
        if selected_exchanges and target_key[0] not in selected_exchanges:
            continue
        if selected_market_types and target_key[1] not in selected_market_types:
            continue
        if selected_symbols and target_key[2] not in selected_symbols:
            continue

        base_asset = TARGET_INSTRUMENTS.get(target_key)
        if base_asset is None:
            ignored_instruments[target_key] = "not_in_target_instruments"
            continue

        key = (rec["exchange"], rec["market_type"], rec["feed"], rec["symbol"])
        if key not in grouped:
            date_glob = next(iter(selected_dates)) if len(selected_dates) == 1 else "*"
            parquet_glob = (
                root
                / rec["exchange"]
                / rec["market_type"]
                / rec["feed"]
                / f"symbol={rec['symbol']}"
                / f"date={date_glob}"
                / "hour=*"
                / "*.parquet"
            )
            instrument_id = "__".join(
                [
                    rec["exchange"],
                    rec["market_type"],
                    rec["feed"],
                    rec["symbol"],
                ]
            )

            grouped[key] = {
                "instrument_id": instrument_id,
                "exchange": rec["exchange"],
                "market_type": rec["market_type"],
                "feed": rec["feed"],
                "symbol": rec["symbol"],
                "base_asset": base_asset,
                "group": assign_group(rec["exchange"], rec["market_type"]),
                "parquet_glob": str(parquet_glob),
                "_schema_file": str(path),
            }

        hour_file_counts[(*key, rec["date"], rec["hour"])] += 1

    # Resolve one explicit research clock per instrument. Do not silently use
    # event time when transaction time is absent (or vice versa).
    schema_con = duckdb.connect()
    try:
        for key in list(grouped):
            base = grouped[key]
            schema_file = str(base.pop("_schema_file"))
            columns = get_columns(
                schema_con, build_scan_expr(schema_file, True)
            )
            try:
                timestamp_col = choose_timestamp_basis_col(
                    columns, args.timestamp_basis
                )
            except RuntimeError:
                target_key = (
                    str(base["exchange"]).lower(),
                    str(base["market_type"]).lower(),
                    str(base["symbol"]).upper(),
                )
                ignored_instruments[target_key] = (
                    f"missing_{args.timestamp_basis}_timestamp"
                )
                del grouped[key]
                continue
            base["timestamp_basis"] = args.timestamp_basis
            base["timestamp_col"] = timestamp_col
    finally:
        schema_con.close()

    # The hourly pipeline contract is exactly one Parquet file per instrument
    # hour.  A directory containing many five-second compact files is excluded
    # as one unit so it cannot receive hundreds of times the weight of a normal
    # hour in the downstream per-file quality aggregation.
    hours_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    excluded_hour_rows: list[dict[str, object]] = []
    for hour_key, file_count in sorted(hour_file_counts.items()):
        exchange, market_type, feed, symbol, date, hour = hour_key
        instrument_hour_key = (exchange, market_type, feed, symbol)
        if instrument_hour_key not in grouped:
            continue
        if file_count == 1:
            hours_by_key[instrument_hour_key].add(f"{date} {hour}")
            continue
        excluded_hour_rows.append(
            {
                "exchange": exchange,
                "market_type": market_type,
                "feed": feed,
                "symbol": symbol,
                "date": date,
                "hour": hour,
                "file_count": file_count,
                "reason": "multiple_parquet_files_in_hour",
            }
        )

    rows = []
    for key, base in sorted(grouped.items()):
        hours = sorted(hours_by_key[key])
        if not hours:
            continue
        rows.append({
            **base,
            "first_hour": hours[0] if hours else "",
            "last_hour": hours[-1] if hours else "",
            "n_hours": len(hours),
        })

    universe_csv = out_dir / "universe.csv"
    excluded_hours_csv = out_dir / "excluded_hours.csv"
    ignored_instruments_csv = out_dir / "ignored_instruments.csv"
    write_csv_rows(
        excluded_hours_csv,
        excluded_hour_rows,
        [
            "exchange",
            "market_type",
            "feed",
            "symbol",
            "date",
            "hour",
            "file_count",
            "reason",
        ],
    )
    write_csv_rows(
        ignored_instruments_csv,
        [
            {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "reason": reason,
            }
            for (exchange, market_type, symbol), reason in sorted(
                ignored_instruments.items()
            )
        ],
        ["exchange", "market_type", "symbol", "reason"],
    )
    write_csv_rows(
        universe_csv,
        rows,
        [
            "instrument_id",
            "exchange",
            "market_type",
            "feed",
            "symbol",
            "base_asset",
            "group",
            "parquet_glob",
            "timestamp_basis",
            "timestamp_col",
            "first_hour",
            "last_hour",
            "n_hours",
        ],
    )

    group_a_rows = [row for row in rows if row["group"] == "A"]
    group_b_rows = [row for row in rows if row["group"] == "B"]
    pair_rows = [
        {
            "base_asset": a["base_asset"],
            "x_instrument_id": a["instrument_id"],
            "x_exchange": a["exchange"],
            "x_market_type": a["market_type"],
            "x_symbol": a["symbol"],
            "y_instrument_id": b["instrument_id"],
            "y_exchange": b["exchange"],
            "y_market_type": b["market_type"],
            "y_symbol": b["symbol"],
        }
        for a in group_a_rows
        for b in group_b_rows
        if a["base_asset"] == b["base_asset"]
    ]

    pairs_csv = out_dir / "pairs.csv"
    write_csv_rows(
        pairs_csv,
        pair_rows,
        [
            "base_asset",
            "x_instrument_id",
            "x_exchange",
            "x_market_type",
            "x_symbol",
            "y_instrument_id",
            "y_exchange",
            "y_market_type",
            "y_symbol",
        ],
    )

    write_json(
        out_dir / "task_config.json",
        {
            "raw_root": str(root),
            "feed": args.feed,
            "timestamp_basis": args.timestamp_basis,
            "universe_csv": str(universe_csv),
            "excluded_hours_csv": str(excluded_hours_csv),
            "ignored_instruments_csv": str(ignored_instruments_csv),
            "pairs_csv": str(pairs_csv),
            "quality_csv": str(out_dir / "instrument_quality.csv"),
            "interval_cache_dir": str(out_dir / "interval_cache"),
            "pair_components_dir": str(out_dir / "hy_pair_components"),
            "summary_dir": str(out_dir / "hy_pair_summary"),
            "group_a": {
                "exchange": "binance",
                "market_type": "perp",
            },
            "group_b": {
                "rule": "all instruments except Binance perp",
            },
            "instrument_filter": {
                "rule": "only instruments in TARGET_INSTRUMENTS",
                "target_count": len(TARGET_INSTRUMENTS),
            },
            "pairing": {
                "rule": "A x B where base_asset is equal",
                "pair_key": "base_asset",
            },
        },
    )

    n_group_a = sum(row["group"] == "A" for row in rows)
    n_group_b = sum(row["group"] == "B" for row in rows)

    print(f"[DONE] instruments={len(rows)}")
    print(f"[DONE] group_A={n_group_a}")
    print(f"[DONE] group_B={n_group_b}")
    print(f"[DONE] eligible_same_asset_pairs={len(pair_rows)}")
    print(f"[DONE] excluded_abnormal_hours={len(excluded_hour_rows)}")
    print(f"[DONE] ignored_instruments={len(ignored_instruments)}")
    print(f"[DONE] wrote {universe_csv}")
    print(f"[DONE] wrote {excluded_hours_csv}")
    print(f"[DONE] wrote {ignored_instruments_csv}")
    print(f"[DONE] wrote {pairs_csv}")
    print(f"[DONE] wrote {out_dir / 'task_config.json'}")


if __name__ == "__main__":
    main()
