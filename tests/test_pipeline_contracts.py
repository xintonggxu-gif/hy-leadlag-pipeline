from __future__ import annotations

import importlib
import math
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime
from pathlib import Path

import numpy as np
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

build_universe = importlib.import_module("01_build_universe")
scan_quality = importlib.import_module("02_scan_quality_low_memory")
build_cache = importlib.import_module("04_build_interval_cache_lowram_fixed")
compute_hy = importlib.import_module("05_compute_pair_hy_lowram_safe_8gb")
common = importlib.import_module("common")


class UniverseContractTests(unittest.TestCase):
    def test_actual_source_inventory_produces_35_instruments_and_43_pairs(self):
        actual = {
            ("binance", "perp"): """
                1000PEPEUSDC 1000PEPEUSDT BTCUSDT DOGEUSDC DOGEUSDT
                ETHUSDC ETHUSDT FARTCOINUSDT SOLUSDC SOLUSDT USDCUSDT
                XRPUSDC XRPUSDT
            """.split(),
            ("binance", "spot"): """
                DOGEUSDC DOGEUSDT ETHUSDC ETHUSDT EURUSDC EURUSDT
                PEPEUSDC PEPEUSDT SOLUSDC SOLUSDT USDCUSDT XRPUSDC XRPUSDT
            """.split(),
            ("bybit", "perp"): """
                1000PEPEPERP 1000PEPEUSDT DOGEPERP DOGEUSDT ETHPERP
                ETHUSDT FARTCOINUSDT SOLPERP SOLUSDT USDCUSDT XRPPERP XRPUSDT
            """.split(),
        }
        rows = []
        for (exchange, market_type), symbols in actual.items():
            for symbol in symbols:
                base_asset = build_universe.TARGET_INSTRUMENTS.get(
                    (exchange, market_type, symbol)
                )
                if base_asset is not None:
                    rows.append(
                        (
                            exchange,
                            market_type,
                            symbol,
                            base_asset,
                            build_universe.assign_group(exchange, market_type),
                        )
                    )

        group_a = [row for row in rows if row[4] == "A"]
        group_b = [row for row in rows if row[4] == "B"]
        pair_count = sum(
            x[3] == y[3]
            for x in group_a
            for y in group_b
        )

        self.assertEqual(len(rows), 35)
        self.assertEqual(len(group_a), 12)
        self.assertEqual(len(group_b), 23)
        self.assertEqual(pair_count, 43)

        transaction_rows = [
            row
            for row in rows
            if not (row[0] == "binance" and row[1] == "spot")
        ]
        transaction_a = [row for row in transaction_rows if row[4] == "A"]
        transaction_b = [row for row in transaction_rows if row[4] == "B"]
        transaction_pair_count = sum(
            x[3] == y[3] for x in transaction_a for y in transaction_b
        )
        self.assertEqual(len(transaction_rows), 24)
        self.assertEqual(transaction_pair_count, 22)

    def test_real_bybit_perp_symbols_are_explicitly_mapped(self):
        expected = {
            "1000PEPEPERP": "1000PEPE",
            "DOGEPERP": "DOGE",
            "ETHPERP": "ETH",
            "SOLPERP": "SOL",
            "XRPPERP": "XRP",
            "USDCUSDT": "USDC",
        }
        for symbol, base_asset in expected.items():
            self.assertEqual(
                build_universe.TARGET_INSTRUMENTS[("bybit", "perp", symbol)],
                base_asset,
            )

    def test_transaction_universe_excludes_spot_without_transaction_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "market_data"
            out = Path(directory) / "run"
            con = duckdb.connect()
            instruments = [
                ("binance", "perp", True),
                ("binance", "spot", False),
                ("bybit", "perp", True),
            ]
            for exchange, market_type, has_transaction in instruments:
                path = (
                    root
                    / exchange
                    / market_type
                    / "bbo"
                    / "symbol=SOLUSDT"
                    / "date=2026-07-01"
                    / "hour=09"
                    / "data.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                transaction_projection = (
                    ", TIMESTAMPTZ '2026-07-01 09:00:00+00' "
                    "AS transaction_time"
                    if has_transaction
                    else ""
                )
                con.execute(
                    f"""
                    COPY (
                        SELECT
                            TIMESTAMPTZ '2026-07-01 09:00:00+00'
                                AS recv_time_ns,
                            TIMESTAMPTZ '2026-07-01 09:00:00+00'
                                AS event_time
                            {transaction_projection},
                            'SOLUSDT' AS symbol,
                            1::BIGINT AS update_id,
                            100.0::DOUBLE AS bid_price,
                            1.0::DOUBLE AS bid_qty,
                            100.1::DOUBLE AS ask_price,
                            1.0::DOUBLE AS ask_qty
                    ) TO ? (FORMAT PARQUET)
                    """,
                    [str(path)],
                )
            con.close()

            argv = [
                "01_build_universe.py",
                "--root",
                str(root),
                "--out",
                str(out),
                "--timestamp-basis",
                "transaction",
            ]
            with mock.patch.object(sys, "argv", argv):
                build_universe.main()

            import csv

            with (out / "universe.csv").open(newline="") as handle:
                universe = list(csv.DictReader(handle))
            with (out / "pairs.csv").open(newline="") as handle:
                pairs = list(csv.DictReader(handle))
            with (out / "ignored_instruments.csv").open(newline="") as handle:
                ignored = list(csv.DictReader(handle))

            self.assertEqual(len(universe), 2)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(
                {(row["exchange"], row["market_type"]) for row in universe},
                {("binance", "perp"), ("bybit", "perp")},
            )
            self.assertEqual(
                ignored,
                [
                    {
                        "exchange": "binance",
                        "market_type": "spot",
                        "symbol": "SOLUSDT",
                        "reason": "missing_transaction_timestamp",
                    }
                ],
            )


class TimestampContractTests(unittest.TestCase):
    def test_timestamp_columns_are_not_treated_as_integer_epochs(self):
        self.assertEqual(
            common.build_ts_expr("recv_time_ns", "timestamp"),
            'CAST("recv_time_ns" AS TIMESTAMP)',
        )
        self.assertEqual(
            scan_quality.build_ts_expr("recv_time_ns", "timestamp"),
            'TRY_CAST("recv_time_ns" AS TIMESTAMP)',
        )

    def test_timestamp_basis_never_falls_back_to_another_clock(self):
        columns = {"recv_time_ns", "event_time"}
        self.assertEqual(
            common.choose_timestamp_basis_col(columns, "event"),
            "event_time",
        )
        with self.assertRaisesRegex(RuntimeError, "transaction"):
            common.choose_timestamp_basis_col(columns, "transaction")

    def test_dedup_order_prefers_exchange_sequence_then_receive_time(self):
        order = common.order_cols_for_dedup(
            {
                "cross_seq",
                "update_id",
                "event_time",
                "recv_time_ns",
                "bid_price",
            }
        )
        self.assertLess(order.index('"cross_seq"'), order.index('"update_id"'))
        self.assertLess(order.index('"update_id"'), order.index('"event_time"'))
        self.assertLess(order.index('"event_time"'), order.index('"recv_time_ns"'))

    def test_event_time_keep_last_is_used_by_quality_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "intervals.parquet"
            con = duckdb.connect()
            con.execute("SET TimeZone = 'UTC'")
            con.execute(
                """
                COPY (
                    SELECT * FROM (VALUES
                        (TIMESTAMPTZ '2026-07-01 09:00:00+00',
                         1::BIGINT, 'ETHUSDT',
                         100.0::DOUBLE, 1.0::DOUBLE,
                         100.02::DOUBLE, 1.0::DOUBLE),
                        (TIMESTAMPTZ '2026-07-01 09:00:00+00',
                         2::BIGINT, 'ETHUSDT',
                         101.0::DOUBLE, 1.0::DOUBLE,
                         101.02::DOUBLE, 1.0::DOUBLE),
                        (TIMESTAMPTZ '2026-07-01 09:00:00.010+00',
                         3::BIGINT, 'ETHUSDT',
                         103.0::DOUBLE, 1.0::DOUBLE,
                         103.02::DOUBLE, 1.0::DOUBLE)
                    ) values_table(
                        event_time, update_id, symbol,
                        bid_price, bid_qty, ask_price, ask_qty
                    )
                ) TO ? (FORMAT PARQUET)
                """,
                [str(source)],
            )
            columns = scan_quality.get_columns(con, str(source))
            dedup_order = common.order_cols_for_dedup(set(columns))
            ts_expr = scan_quality.build_ts_expr("event_time", "timestamp")
            row = {"symbol": "ETHUSDT"}
            base = scan_quality.scan_base_stats(
                con,
                row,
                str(source),
                ts_expr,
                "TRUE",
                "mid",
                10.0,
                None,
                None,
                dedup_order,
                "keep-last",
            )
            timing = scan_quality.scan_clean_time_stats(
                con,
                row,
                str(source),
                ts_expr,
                "TRUE",
                "mid",
                10.0,
                None,
                None,
                dedup_order,
                "keep-last",
            )
            self.assertEqual(base["total_rows"], 3)
            self.assertEqual(base["selected_rows"], 2)
            self.assertEqual(base["deduplicated_rows"], 1)
            self.assertEqual(base["duplicate_ts"], 1)
            self.assertEqual(base["clean_rows"], 2)
            self.assertEqual(timing["ret_count"], 1)

            scan = build_cache.scan_expr_for_files([str(source)])
            cache_sql = build_cache.build_copy_sql(
                row={
                    "exchange": "binance",
                    "market_type": "perp",
                    "symbol": "ETHUSDT",
                },
                scan=scan,
                ts_expr=common.build_ts_expr("event_time", "timestamp"),
                dedup_order_by=dedup_order,
                duplicate_ts_policy="keep-last",
                price_mode="mid",
                max_interval_ms=100.0,
                max_spread_bps=10.0,
                min_interval_ms=0.0,
                scan_start=datetime(2026, 7, 1, 8, 59, 59),
                chunk_start=datetime(2026, 7, 1, 9),
                chunk_end=datetime(2026, 7, 1, 10),
                excluded_hours=set(),
                drop_zero_returns=False,
                out_file=output,
            )
            con.execute(cache_sql)
            intervals = con.execute(
                "SELECT interval_ms, ret FROM read_parquet(?)", [str(output)]
            ).fetchall()
            self.assertEqual(len(intervals), 1)
            self.assertEqual(intervals[0][0], 10.0)
            expected = math.log(103.01) - math.log(101.01)
            self.assertAlmostEqual(intervals[0][1], expected)
            con.close()


class CacheCoverageContractTests(unittest.TestCase):
    def test_chunk_parser_accepts_old_and_microsecond_names(self):
        old = Path(
            "mid_chunk_start=20260715T000000_"
            "chunk_end=20260715T010000.parquet"
        )
        new = Path(
            "mid_chunk_start=20260715T000000.123000_"
            "chunk_end=20260715T010000.456000.parquet"
        )
        old_span = compute_hy.parse_chunk_span(old)
        new_span = compute_hy.parse_chunk_span(new)
        self.assertIsNotNone(old_span)
        self.assertIsNotNone(new_span)
        assert old_span is not None and new_span is not None
        self.assertEqual(
            old_span[0],
            common.dt_to_ns(datetime(2026, 7, 15, 0, 0, 0)),
        )
        self.assertEqual(
            new_span[0],
            common.dt_to_ns(datetime(2026, 7, 15, 0, 0, 0, 123000)),
        )
        self.assertEqual(
            new_span[1],
            common.dt_to_ns(datetime(2026, 7, 15, 1, 0, 0, 456000)),
        )


class EstimatorContractTests(unittest.TestCase):
    def test_synchronous_identical_intervals_have_unit_diagnostic_corr(self):
        start = np.array([0, 10], dtype=np.int64)
        end = np.array([10, 20], dtype=np.int64)
        returns = np.array([1.0, 2.0], dtype=np.float64)
        cov, var_x, var_y, n_overlap, n_x, n_y = (
            compute_hy.hy_for_lag_boundary_fixed(
                start,
                end,
                returns,
                start,
                end,
                returns,
                0,
                0,
                20,
                analysis_end_ns=20,
            )
        )
        self.assertEqual((cov, var_x, var_y), (5.0, 5.0, 5.0))
        self.assertEqual((n_overlap, n_x, n_y), (2, 2, 2))
        self.assertAlmostEqual(
            compute_hy.corr_or_none(cov, var_x, var_y) or 0.0,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
