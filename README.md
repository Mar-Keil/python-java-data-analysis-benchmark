# python-java-data-analysis-benchmark

Benchmark project for comparing Python data analysis with Polars and Java data analysis with DFLib.

## Overview

This repository contains three main parts:

- `data_gen`: Generates the benchmark datasets as parquet files.
- `python`: Runs the Python benchmark implementation based on Polars.
- `java`: Runs the Java benchmark implementation based on JMH and DFLib.

The benchmark currently covers these operation categories:

- Read
- Filter
- Sort
- Pivot
- GroupCount
- Join
- WriteSort

The configured flight dataset sizes are:

- `125k`
- `500k`
- `2000k`
- `8000k`

## Repository Structure

```text
.
|-- data_gen
|   |-- src
|   `-- out
|-- java
|   |-- src
|   `-- target
|-- latex
|-- python
|   `-- src
`-- run-benchmark.sh
```

## Requirements

- Python `3.14`
- Java `25`
- Maven
- `caffeinate` for the convenience launcher on macOS

## Setup

The dataset generator has its own Python project configuration in
`data_gen/pyproject.toml`. The root launcher bootstraps the shared Python
environment automatically inside `data_gen/.venv`.

In the normal case, no manual Python setup is required before the first run.

## Running the Benchmark

Run the complete benchmark workflow from the repository root:

```bash
./run-benchmark.sh
```

The script performs these steps:

1. Creates `data_gen/.venv` if it does not exist yet.
2. Installs or repairs the Python dependencies from `data_gen/pyproject.toml`.
3. Checks whether all expected datasets already exist in `data_gen/out`.
4. Generates missing datasets with `generate-datasets`.
5. Runs the Python benchmark from `python/src/benchmark/run_benchmark.py`.
6. Packages and runs the Java benchmark from `java`.

If you want to execute the steps manually, you can use:

```bash
cd data_gen
. .venv/bin/activate
generate-datasets
```

```bash
cd /path/to/python-java-data-analysis-benchmark
env PYTHONPATH=python/src data_gen/.venv/bin/python python/src/benchmark/run_benchmark.py
```

```bash
mvn -f java/pom.xml -q -DskipTests package
java -jar java/target/benchmarks.jar
```

## Output

Generated datasets are written to:

- `data_gen/out`

Python benchmark results are written to:

- `python/src/benchmark/out`

Java benchmark results are written to:

- `java/out/csv`

Temporary Java write benchmark output is written to:

- `java/out/jmh-write`

## Notes

- The root launcher currently assumes a macOS environment because it uses
  `caffeinate`. On other systems, run the individual benchmark steps manually.
- The Python benchmark code is source-based and is started through `PYTHONPATH`
  instead of a separate installable Python package.
