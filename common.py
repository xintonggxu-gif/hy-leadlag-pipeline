from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def ident(s: str) -> str:
    return '"' + str(s).replace('"', '""') + '"'


def parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO datetime string and normalize aware datetimes to naive UTC.

    Examples accepted:
      2026-06-26
      2026-06-26T13:00:00
      2026-06-26T13:00:00Z
      2026-06-26T13:00:00+08:00
    """
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def ts_lit(dt: datetime) -> str:
    """DuckDB TIMESTAMP literal in UTC-naive form."""
    return f"TIMESTAMP '{dt.strftime('%Y-%m-%d %H:%M:%S.%f')}'"


def dt_to_ns(dt: datetime) -> int:
    """Convert datetime to epoch nanoseconds using integer arithmetic.

    The previous float-based implementation, int(dt.timestamp() * 1e9), can
    introduce tiny rounding errors. Your market data is microsecond precision,
    so this usually would not change results, but integer arithmetic is cleaner
    and deterministic.

    Naive datetimes are treated as UTC, matching parse_dt()/ns_to_dt().
    """
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    delta = dt_utc - _EPOCH_UTC
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def ns_to_dt(ns: int) -> datetime:
    """Convert epoch nanoseconds to naive UTC datetime.

    Python datetime stores microseconds, so sub-microsecond remainder is dropped.
    This is fine for your current recv_time_us pipeline.
    """
    us = int(ns) // 1_000
    return (_EPOCH_UTC + timedelta(microseconds=us)).replace(tzinfo=None)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_excluded_hours_path(
    universe_path: str | Path,
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Resolve the optional abnormal-hour manifest beside universe.csv.

    An explicit path is strict.  Automatic discovery is backward compatible:
    older runs without excluded_hours.csv simply have no exclusions.
    """
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidate = Path(universe_path).expanduser().resolve().with_name(
        "excluded_hours.csv"
    )
    return candidate if candidate.is_file() else None


def load_excluded_hours(
    path: str | Path | None,
) -> dict[tuple[str, str, str], set[datetime]]:
    """Load excluded UTC hours using only O(number of abnormal hours) RAM."""
    if path is None:
        return {}

    rows = read_csv_rows(path)
    if not rows:
        return {}

    required = {"exchange", "market_type", "symbol", "date", "hour"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"excluded-hours CSV missing columns: {sorted(missing)}"
        )

    result: dict[tuple[str, str, str], set[datetime]] = {}
    for row in rows:
        try:
            hour_number = int(row["hour"])
            if not 0 <= hour_number <= 23:
                raise ValueError
            hour_start = datetime.fromisoformat(
                f"{row['date']} {hour_number:02d}:00:00"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid excluded hour: "
                f"date={row.get('date')!r} hour={row.get('hour')!r}"
            ) from exc

        key = (row["exchange"], row["market_type"], row["symbol"])
        result.setdefault(key, set()).add(hour_start)
    return result


def write_csv_rows(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def build_scan_expr(path: str, hive_partitioning: bool = True) -> str:
    hive = "true" if hive_partitioning else "false"
    return f"read_parquet({sql_quote(path)}, union_by_name=true, hive_partitioning={hive})"


def get_columns(con, scan: str) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    return {r[0] for r in rows}


def choose_timestamp_col(cols: set[str], explicit: str | None = None) -> str:
    """Choose the timestamp column used as the event/receive time.

    The compacted Binance/Bybit files store recv_time_us/recv_time_ns as
    TIMESTAMP WITH TIME ZONE values.  The suffix describes the source field,
    not an integer epoch unit, so these columns must normally be used with
    --ts-unit timestamp.  Column selection stays per instrument because Spot
    and Bybit use recv_time_ns while Binance perpetuals use recv_time_us.
    """
    if explicit:
        if explicit not in cols:
            raise RuntimeError(f"timestamp column not found: {explicit}; available={sorted(cols)}")
        return explicit

    for col in ("recv_time_us", "recv_time_ns", "recv_time", "event_time", "transaction_time"):
        if col in cols:
            return col

    raise RuntimeError(f"Cannot infer timestamp column; available={sorted(cols)}")


def choose_timestamp_basis_col(
    cols: set[str],
    basis: str,
    explicit: str | None = None,
) -> str:
    """Resolve a timestamp basis without silently falling back to another clock."""
    if explicit:
        if explicit not in cols:
            raise RuntimeError(
                f"timestamp column not found: {explicit}; available={sorted(cols)}"
            )
        return explicit

    if basis == "receive":
        for col in (
            "recv_time_us",
            "recv_time_ns",
            "recv_time_ms",
            "recv_time",
            "receive_time",
            "receive_ts",
            "received_ts",
            "recv_ts",
        ):
            if col in cols:
                return col
    elif basis == "event":
        for col in ("event_time", "event_ts", "E"):
            if col in cols:
                return col
    elif basis == "transaction":
        for col in ("transaction_time", "transaction_ts", "T"):
            if col in cols:
                return col
    elif basis == "custom":
        raise RuntimeError("custom timestamp basis requires --timestamp-col")
    else:
        raise ValueError(f"unsupported timestamp basis: {basis!r}")

    raise RuntimeError(
        f"no {basis!r} timestamp column found; available={sorted(cols)}"
    )


def build_ts_expr(timestamp_col: str, ts_unit: str) -> str:
    c = ident(timestamp_col)
    if ts_unit == "timestamp":
        return f"CAST({c} AS TIMESTAMP)"
    if ts_unit == "ns":
        return f"make_timestamp_ns(CAST({c} AS BIGINT))"
    if ts_unit == "us":
        return f"make_timestamp(CAST({c} AS BIGINT))"
    if ts_unit == "ms":
        return f"to_timestamp(CAST({c} AS DOUBLE) / 1000.0)::TIMESTAMP"
    raise ValueError(f"Unsupported ts_unit: {ts_unit}")


def order_cols_for_dedup(cols: set[str]) -> str:
    """Return a deterministic ORDER BY clause for same-timestamp BBO dedup.

    Same recv_time can contain multiple quotes. We prefer the latest exchange
    sequence/update fields when present. Remaining fields are deterministic
    tie-breakers; they matter only when all stronger ordering columns tie.
    """
    pieces: list[str] = []

    # Exchange/order-book sequence fields: larger usually means later.
    for col in (
        "cross_seq",
        "seq",
        "sequence",
        "update_id",
        "u",
        "last_update_id",
    ):
        if col in cols:
            pieces.append(f"CAST({ident(col)} AS BIGINT) DESC NULLS LAST")

    # Time-like fields: use as secondary order when available. Do not cast here;
    # they may already be TIMESTAMP, BIGINT, or another orderable type.
    for col in ("transaction_time", "event_time", "T", "E"):
        if col in cols:
            pieces.append(f"{ident(col)} DESC NULLS LAST")

    # If exchange sequence fields tie or are missing, prefer the update that
    # arrived last at the collector.  The compacted receive columns are typed
    # timestamps despite their historical suffixes.
    for col in ("recv_time_us", "recv_time_ns", "recv_time"):
        if col in cols:
            pieces.append(f"{ident(col)} DESC NULLS LAST")

    # Stable final tie-breakers. These should not normally decide rows, but they
    # make ROW_NUMBER deterministic when duplicate timestamps are otherwise tied.
    for col in ("bid_price", "ask_price", "bid_qty", "ask_qty"):
        if col in cols:
            pieces.append(f"{ident(col)} DESC NULLS LAST")

    # ts is always available in the query where this ORDER BY is used.
    pieces.append("ts DESC")
    return ", ".join(pieces)


def instrument_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["exchange"], row["market_type"], row["symbol"]


def interval_glob(
    interval_root: str | Path,
    exchange: str,
    market_type: str,
    symbol: str,
    price_mode: str | None = None,
) -> str:
    """Glob interval-cache parquet files for one instrument.

    Backward compatible with the old layout:
      root/exchange=.../market_type=.../symbol=.../**/*.parquet

    Also supports the safer new layout when price_mode is provided:
      root/exchange=.../market_type=.../symbol=.../price_mode=mid/**/*.parquet
    """
    root = Path(interval_root) / f"exchange={exchange}" / f"market_type={market_type}" / f"symbol={symbol}"
    if price_mode:
        root = root / f"price_mode={price_mode}"
    return str(root / "**" / "*.parquet")
