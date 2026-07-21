from __future__ import annotations

import csv
import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb


latency_audit = importlib.import_module("07_audit_receive_event_latency")


class LatencyAuditTests(unittest.TestCase):
    @staticmethod
    def _write_source(
        path: Path,
        receive_column: str,
        symbol: str,
        latency_ms: int,
    ) -> None:
        path.parent.mkdir(parents=True)
        con = duckdb.connect()
        con.execute(
            f"""
                CREATE TABLE source AS
                SELECT
                    TIMESTAMPTZ '2026-07-01 09:00:00+00'
                        + INTERVAL ({latency_ms}) MILLISECOND
                        AS {receive_column},
                    TIMESTAMPTZ '2026-07-01 09:00:00+00' AS event_time,
                    '{symbol}' AS symbol
            """
        )
        con.execute(f"COPY source TO '{path}' (FORMAT PARQUET)")
        con.close()

    def test_receive_column_matches_compacted_schemas(self):
        self.assertEqual(
            latency_audit.choose_receive_column(
                {"recv_time_us", "event_time", "symbol"}
            ),
            "recv_time_us",
        )
        self.assertEqual(
            latency_audit.choose_receive_column(
                {"recv_time_ns", "event_time", "symbol"}
            ),
            "recv_time_ns",
        )
        self.assertEqual(
            latency_audit.parse_venues("binance/perp,binance/spot,bybit/perp"),
            (
                ("binance", "perp"),
                ("binance", "spot"),
                ("bybit", "perp"),
            ),
        )

    def test_scan_hour_file_reports_receive_minus_event_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "symbol=TESTUSDT"
                / "date=2026-07-01"
                / "hour=09"
                / "part.parquet"
            )
            path.parent.mkdir(parents=True)
            con = duckdb.connect()
            con.execute(
                """
                CREATE TABLE source AS
                SELECT * FROM (VALUES
                    (TIMESTAMPTZ '2026-07-01 09:00:00.010+00', TIMESTAMPTZ '2026-07-01 09:00:00.000+00', 'TESTUSDT'),
                    (TIMESTAMPTZ '2026-07-01 09:00:01.020+00', TIMESTAMPTZ '2026-07-01 09:00:01.000+00', 'TESTUSDT'),
                    (TIMESTAMPTZ '2026-07-01 09:00:01.995+00', TIMESTAMPTZ '2026-07-01 09:00:02.000+00', 'TESTUSDT'),
                    (TIMESTAMPTZ '2026-07-01 09:00:03.000+00', NULL, 'TESTUSDT')
                ) AS t(recv_time_ns, event_time, symbol)
            """
            )
            con.execute(f"COPY source TO '{path}' (FORMAT PARQUET)")
            result = latency_audit.scan_hour_file(
                con,
                {
                    "exchange": "bybit",
                    "market_type": "perp",
                    "symbol": "TESTUSDT",
                },
                path,
                "recv_time_ns",
                "event_time",
                datetime(2026, 7, 1, 9),
                datetime(2026, 7, 1, 10),
            )
            con.close()

        self.assertEqual(result["total_rows"], 4)
        self.assertEqual(result["valid_latency_rows"], 3)
        self.assertEqual(result["event_null_or_invalid_rows"], 1)
        self.assertEqual(result["negative_latency_rows"], 1)
        self.assertAlmostEqual(float(result["latency_min_ms"]), -5.0)
        self.assertAlmostEqual(float(result["latency_p50_ms"]), 10.0)
        self.assertAlmostEqual(float(result["latency_max_ms"]), 20.0)

    def test_summary_uses_hour_equal_medians_and_weighted_mean(self):
        rows = [
            {
                "total_rows": "100",
                "valid_latency_rows": "100",
                "negative_latency_rows": "0",
                "latency_gt_100ms_rows": "0",
                "latency_mean_ms": "10",
                "latency_p01_ms": "5",
                "latency_p50_ms": "10",
                "latency_p90_ms": "20",
                "latency_p95_ms": "25",
                "latency_p99_ms": "30",
                "latency_p999_ms": "40",
            },
            {
                "total_rows": "300",
                "valid_latency_rows": "300",
                "negative_latency_rows": "30",
                "latency_gt_100ms_rows": "60",
                "latency_mean_ms": "30",
                "latency_p01_ms": "15",
                "latency_p50_ms": "30",
                "latency_p90_ms": "40",
                "latency_p95_ms": "45",
                "latency_p99_ms": "50",
                "latency_p999_ms": "60",
            },
        ]
        summary = latency_audit.summarize_rows(rows)
        self.assertEqual(summary["n_hours"], 2)
        self.assertEqual(summary["total_rows"], 400)
        self.assertAlmostEqual(float(summary["event_weighted_mean_ms"]), 25.0)
        self.assertAlmostEqual(float(summary["median_hour_p50_ms"]), 20.0)
        self.assertAlmostEqual(float(summary["negative_latency_ratio"]), 0.075)
        self.assertAlmostEqual(float(summary["latency_gt_100ms_ratio"]), 0.15)

    def test_main_writes_exchange_comparison_and_safely_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binance_path = (
                root
                / "binance"
                / "perp"
                / "bbo"
                / "symbol=TESTUSDT"
                / "date=2026-07-01"
                / "hour=09"
                / "part.parquet"
            )
            binance_spot_path = (
                root
                / "binance"
                / "spot"
                / "bbo"
                / "symbol=TESTUSDT"
                / "date=2026-07-01"
                / "hour=09"
                / "part.parquet"
            )
            bybit_path = (
                root
                / "bybit"
                / "perp"
                / "bbo"
                / "symbol=TESTUSDT"
                / "date=2026-07-01"
                / "hour=09"
                / "part.parquet"
            )
            self._write_source(binance_path, "recv_time_us", "TESTUSDT", 10)
            self._write_source(binance_spot_path, "recv_time_ns", "TESTUSDT", 30)
            self._write_source(bybit_path, "recv_time_ns", "TESTUSDT", 80)

            universe = root / "universe.csv"
            with universe.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "exchange",
                        "market_type",
                        "symbol",
                        "base_asset",
                        "parquet_glob",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "exchange": "binance",
                        "market_type": "spot",
                        "symbol": "TESTUSDT",
                        "base_asset": "TEST",
                        "parquet_glob": str(binance_spot_path),
                    }
                )
                writer.writerow(
                    {
                        "exchange": "binance",
                        "market_type": "perp",
                        "symbol": "TESTUSDT",
                        "base_asset": "TEST",
                        "parquet_glob": str(binance_path),
                    }
                )
                writer.writerow(
                    {
                        "exchange": "bybit",
                        "market_type": "perp",
                        "symbol": "TESTUSDT",
                        "base_asset": "TEST",
                        "parquet_glob": str(bybit_path),
                    }
                )

            out_dir = root / "audit"
            argv = [
                "07_audit_receive_event_latency.py",
                "--universe",
                str(universe),
                "--out",
                str(out_dir),
                "--start",
                "2026-07-01T09:00:00Z",
                "--end",
                "2026-07-01T10:00:00Z",
                "--temp-dir",
                str(root / "duckdb_tmp"),
                "--min-free-disk-gb",
                "0",
                "--reopen-every",
                "1",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                latency_audit.main()

            with (out_dir / "receive_event_latency_hourly.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                hourly = list(csv.DictReader(handle))
            report = json.loads(
                (out_dir / "receive_event_latency_report.json").read_text()
            )
            self.assertEqual(len(hourly), 3)
            self.assertAlmostEqual(
                report["venue_comparisons_ms"]["bybit/perp - binance/perp"][
                    "median_hour_p50_ms"
                ],
                70.0,
            )
            self.assertAlmostEqual(
                report["venue_comparisons_ms"]["bybit/perp - binance/spot"][
                    "median_hour_p50_ms"
                ],
                50.0,
            )
            self.assertAlmostEqual(
                report["venue_comparisons_by_base_asset_ms"][
                    "binance/perp - binance/spot"
                ]["TEST"]["median_hour_p50_ms"],
                -20.0,
            )

            # The same configuration may resume and must not duplicate rows.
            with patch.object(sys, "argv", argv + ["--resume"]), redirect_stdout(
                io.StringIO()
            ):
                latency_audit.main()
            with (out_dir / "receive_event_latency_hourly.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                resumed = list(csv.DictReader(handle))
            self.assertEqual(len(resumed), 3)


if __name__ == "__main__":
    unittest.main()
