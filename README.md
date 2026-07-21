# Low-Memory HY Lead-Lag Pipeline

This repository contains the research pipeline I use to study short-horizon lead-lag relationships in asynchronous BBO data.

The pipeline cleans hourly Parquet files, estimates instrument-specific data-quality thresholds, builds event-time return intervals, computes lagged Hayashi-Yoshida covariance, and aggregates the results across time windows.

It is designed to run on a machine with approximately 8 GB of RAM.

## Research question

Fixed-grid correlation is a useful baseline, but it can be misleading when two markets update at different rates. Previous-tick filling may align new quotes from one venue with stale quotes from another venue.

This project therefore uses event-time return intervals and Hayashi-Yoshida covariance as the main estimator. The implementation also controls for long stale intervals, abnormal spreads, chunk boundaries, and inconsistent cache configurations.

The current analysis is pairwise. It is not intended to estimate a full covariance matrix.

## Input layout

The pipeline expects compacted hourly BBO data with one Parquet file per instrument-hour:

```text
market_data_compacted/
  {exchange}/
    {market_type}/
      bbo/
        symbol={SYMBOL}/
          date={YYYY-MM-DD}/
            hour={HH}/
              data.parquet
```

Hours containing multiple compacted files are recorded as abnormal and excluded from downstream stages.

Raw market data is not included in this repository.

## Pipeline

The scripts are intended to run in numerical order.

### 1. Build the instrument universe

`01_build_universe.py` scans the partitioned Parquet tree and creates:

- `universe.csv`
- `excluded_hours.csv`
- `task_config.json`

The current grouping rule is:

- Group A: Binance perpetual instruments
- Group B: other supported instruments

Pair construction matches instruments by normalized base asset unless explicitly overridden.

### 2. Scan data quality

`02_scan_quality_low_memory.py` processes one hourly file at a time and records:

- duplicate and backward timestamps
- crossed or locked quotes
- non-positive prices and quantities
- update-gap quantiles
- spread quantiles
- zero-return ratio

The file-by-file design keeps memory usage bounded and allows interrupted scans to resume.

### 3. Select instrument-level filters

`03_aggregate_quality.py` aggregates the hourly statistics and calculates recommended limits for:

- maximum return-interval length
- maximum spread

These limits are instrument-specific and are used when building the interval cache.

### 4. Build the return-interval cache

`04_build_interval_cache_lowram_fixed.py` converts cleaned BBO observations into event-time log-return intervals.

The default price proxy is mid-price. Microprice can be selected as a robustness check.

Each interval contains a start timestamp, end timestamp, log return, interval length, and filtering metadata. A lookback buffer is used at chunk boundaries so that the first valid return in each chunk is not lost.

### 5. Compute lagged HY components

`05_compute_pair_hy_lowram_safe_8gb.py` computes Hayashi-Yoshida covariance components for each pair, time window, and lag.

The implementation includes explicit limits for:

- DuckDB memory
- NumPy array size
- temporary disk usage
- number of pairs
- number of cache files
- estimated work per window
- output rows
- runtime

Pair jobs can be divided into shards with `--pair-offset` and `--pair-limit`.

### 6. Summarize results

`06_summarize_pair_hy_best.py` validates and aggregates the component files into:

- lag-level summaries
- best absolute-correlation summaries
- best positive-correlation summaries

Configuration hashes and success markers are checked before different component files are combined.

### 7. Audit receive-time latency

`07_audit_receive_event_latency.py` measures local receive time minus exchange
event time for Binance perpetual, Binance spot, and Bybit perpetual BBO files.
It scans one hourly Parquet file at a time, supports safe resume, and writes
hourly, daily, instrument, venue, and venue-by-base-asset summaries. The report
compares Bybit perpetual against both Binance perpetual and Binance spot, as
well as Binance perpetual against Binance spot, overall and for every shared
base asset. This helps distinguish an apparent venue lead-lag from
collector/network delay without confounding the comparison by venue or asset
mix.

Example:

```bash
python 07_audit_receive_event_latency.py \
  --universe ./run/universe.csv \
  --out ./run/receive_event_latency_audit \
  --start 2026-07-01T09:00:00Z \
  --end 2026-07-16T00:00:00Z \
  --event-column event_time \
  --threads 1 \
  --memory-limit 2GB \
  --temp-dir ./run/duckdb_tmp
```

The compacted `recv_time_us` and `recv_time_ns` columns are timestamps, not
integer epochs. Positive latency means the local collector received the BBO
after the exchange event timestamp. Exchange-level quantiles use the median of
hourly per-event quantiles so that each instrument-hour has equal weight.

Only the methodology and output schema are documented here; raw market data, instrument identifiers, and empirical findings are intentionally excluded.

## Installation

Python 3.11 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The current dependencies are:

```text
duckdb
numpy
```

## Minimal run example

Run the following commands from the repository root.

```bash
mkdir -p run
mkdir -p duckdb_tmp
```

Build the universe:

```bash
python 01_build_universe.py \
  --root /path/to/market_data_compacted \
  --out ./run
```

Scan hourly data quality:

```bash
python 02_scan_quality_low_memory.py \
  --universe ./run/universe.csv \
  --partial-out ./run/instrument_quality_hourly.csv \
  --start 2026-06-26T00:00:00Z \
  --end 2026-07-10T00:00:00Z \
  --price-mode mid \
  --threads 1 \
  --memory-limit 2GB \
  --temp-dir ./duckdb_tmp
```

Aggregate the quality results:

```bash
python 03_aggregate_quality.py \
  --partial ./run/instrument_quality_hourly.csv \
  --out ./run/instrument_quality.csv
```

Build the interval cache:

```bash
python 04_build_interval_cache_lowram_fixed.py \
  --universe ./run/universe.csv \
  --quality ./run/instrument_quality.csv \
  --out ./run/interval_cache \
  --start 2026-06-26T00:00:00Z \
  --end 2026-07-10T00:00:00Z \
  --price-mode mid \
  --chunk-minutes 60 \
  --threads 1 \
  --memory-limit 2GB \
  --temp-dir ./duckdb_tmp
```

Compute one pair shard:

```bash
python 05_compute_pair_hy_lowram_safe_8gb.py \
  --universe ./run/universe.csv \
  --interval-root ./run/interval_cache \
  --out ./run/hy_pair_components/shard_000.parquet \
  --start 2026-06-26T00:00:00Z \
  --end 2026-07-10T00:00:00Z \
  --price-mode mid \
  --window-hours 2 \
  --max-lag-ms 100 \
  --lag-step-ms 10 \
  --pair-offset 0 \
  --pair-limit 100 \
  --threads 1 \
  --memory-limit 2GB \
  --temp-dir ./duckdb_tmp
```

Summarize the component files:

```bash
python 06_summarize_pair_hy_best.py \
  --components "./run/hy_pair_components/*.parquet" \
  --out ./run/hy_pair_summary \
  --expected-pairs 0 \
  --min-overlap 100 \
  --threads 1 \
  --memory-limit 2GB \
  --temp-dir ./duckdb_tmp
```

## Lag convention

A positive `lag_ms` means that the X instrument leads the Y instrument by approximately that amount of time.

For example, a peak at `lag_ms = 30` means that shifting X forward by 30 milliseconds produces the strongest measured alignment with Y.

## Interpretation

The reported correlation is a diagnostic measure of lagged co-movement. It should not be interpreted as evidence of causality or as a trading strategy on its own.

A candidate relationship should also be checked across:

- different dates and intraday windows
- mid-price and microprice
- alternative interval caps
- different lag ranges
- grid-based baselines
- skip-sampled or pre-averaged estimators

Peaks at the edge of the lag search range, isolated single-lag spikes, and results supported by very few overlaps require additional investigation.

## Repository status

Implemented:

- hourly universe construction
- data-quality scanning
- adaptive interval and spread caps
- boundary-aware interval cache construction
- pairwise lagged HY computation
- low-memory and disk-safety controls
- configuration validation
- multi-window result aggregation

Planned:

- synthetic reproducibility example
- automated estimator tests
- grid-correlation comparison
- mid-price versus microprice report
- skip-sampled and pre-averaged HY robustness checks
- result visualizations