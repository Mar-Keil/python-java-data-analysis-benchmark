from benchmark.default.default_values import AIRLINES_INPUT_PATH
from benchmark.default.default_values import POLARS_OUT_DIR
from benchmark.default.print_csv import PrintCSV
from benchmark.run.benchmark_operation import benchmark_operation
from benchmark.run.benchmark_read import benchmark_read
from benchmark.run.benchmark_write import benchmark_write
from logic import polars_logic

"""Serves as the entry point for executing the configured benchmark operations."""

class BenchmarkRunner:
    def run(self) -> None:
        print_csv = PrintCSV(POLARS_OUT_DIR)
        benchmark_read(print_csv, polars_logic.read_parquet)
        for operation in ("filter", "sort", "pivot", "group_count", "join"):
            other_dataset = None

            if operation == "join":
                other_dataset = polars_logic.read_parquet(AIRLINES_INPUT_PATH)

            benchmark_operation(
                print_csv,
                polars_logic.read_parquet,
                getattr(polars_logic, f"{operation}_dataset"),
                _method_name(operation),
                other_dataset=other_dataset,
            )

        benchmark_write(
            print_csv,
            POLARS_OUT_DIR,
            polars_logic.read_parquet,
            polars_logic.write_parquet,
        )

def _method_name(operation: str) -> str:
    return "".join(part.capitalize() for part in operation.split("_"))


def main() -> None:
    BenchmarkRunner().run()


if __name__ == "__main__":
    main()
