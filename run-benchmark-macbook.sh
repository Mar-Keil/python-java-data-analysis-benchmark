#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_GEN_DIR="$ROOT_DIR/data_gen"
DATA_OUT_DIR="$DATA_GEN_DIR/out"
DATA_GEN_VENV_BIN_DIR="$DATA_GEN_DIR/.venv/bin"
DATASET_GENERATOR="$DATA_GEN_VENV_BIN_DIR/generate-datasets"
JAVA_DIR="$ROOT_DIR/java"
BENCHMARK_JAR="$JAVA_DIR/target/benchmarks.jar"

EXPECTED_DATASETS=(
  "airlines.parquet"
  "31.25kFlights.parquet"
  "125kFlights.parquet"
  "500kFlights.parquet"
  "2000kFlights.parquet"
  "8000kFlights.parquet"
)

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name was not found."
    echo "Install $command_name and make it available on PATH, then rerun this script."
    exit 1
  fi
}

has_all_datasets() {
  local dataset_name

  for dataset_name in "${EXPECTED_DATASETS[@]}"; do
    if [[ ! -f "$DATA_OUT_DIR/$dataset_name" ]]; then
      return 1
    fi
  done

  return 0
}

print_dataset_status() {
  local dataset_name

  echo "Dataset status:"
  for dataset_name in "${EXPECTED_DATASETS[@]}"; do
    if [[ -f "$DATA_OUT_DIR/$dataset_name" ]]; then
      echo "  present: $dataset_name"
    else
      echo "  missing: $dataset_name"
    fi
  done
}

generate_datasets_if_needed() {
  if has_all_datasets; then
    echo "All benchmark datasets are already present."
    return 0
  fi

  print_dataset_status

  if [[ ! -x "$DATASET_GENERATOR" ]]; then
    echo "Dataset generator was not found at: $DATASET_GENERATOR"
    echo "Create the data_gen virtual environment and install the generator dependencies first."
    exit 1
  fi

  echo "Generating missing benchmark datasets..."
  "$DATASET_GENERATOR"

  if ! has_all_datasets; then
    echo "Dataset generation finished, but not all expected datasets were created."
    print_dataset_status
    exit 1
  fi
}

run_java_benchmark() {
  require_command mvn
  require_command java

  echo "Packaging Java benchmark..."
  mvn -f "$JAVA_DIR/pom.xml" -q -DskipTests package

  if [[ ! -f "$BENCHMARK_JAR" ]]; then
    echo "Benchmark JAR was not created: $BENCHMARK_JAR"
    exit 1
  fi

  echo "Running Java benchmark..."
  java -jar "$BENCHMARK_JAR"
}

main() {
  generate_datasets_if_needed
  run_java_benchmark
}

main "$@"
