from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parents[1]
BENCHMARKING_DIR = BASE_DIR / "benchmark"

DATA_GEN_OUT_DIR = BASE_DIR / "data_gen" / "out"
BENCHMARK_OUT_DIR = BENCHMARKING_DIR / "out"
POLARS_OUT_DIR = BENCHMARK_OUT_DIR / "polars_results"

BENCHMARK_ITERATIONS = 10
BENCHMARK_DURATION_SECONDS = 10

AIRLINES_INPUT_PATH = DATA_GEN_OUT_DIR / "airlines.parquet"

PARAM = (
    DATA_GEN_OUT_DIR / "31.25kFlights.parquet",
    DATA_GEN_OUT_DIR / "125kFlights.parquet",
    DATA_GEN_OUT_DIR / "500kFlights.parquet",
    DATA_GEN_OUT_DIR / "2000kFlights.parquet",
    DATA_GEN_OUT_DIR / "8000kFlights.parquet",
)
