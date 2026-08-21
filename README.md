# Predictive maintenance

This repository contains reproducible anomaly-detection experiments for industrial sensor data. The original notebooks are retained for exploration, while the supported command-line scripts and reusable package live under `scripts/` and `src/`.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```sh
uv sync
```

## Run an experiment

The input CSV must contain a `Label` column and may contain a timestamp column. Labels use `0` for normal observations and `1` for faults.

```sh
uv run python scripts/isolation_forest.py data/features.csv --output results/run
```

The two notebooks document the historical autoencoder and isolation-forest experiments. Their script equivalents are intentionally small, deterministic entry points that can be used in automation.

## Quality checks

```sh
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Data and results

Raw datasets are not committed because the notebooks reference local and hosted paths. Existing plots and metric snapshots in `results/` are retained as historical outputs.
