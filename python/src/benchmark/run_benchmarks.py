from benchmark.default.default_values import AIRLINES_INPUT_PATH
from benchmark.default.default_values import POLARS_OUT_DIR
from benchmark.default.print_csv import PrintCSV
from benchmark.run.benchmark_operation import benchmark_operation
from benchmark.run.benchmark_read import benchmark_read
from logic import polars_logic


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
                POLARS_OUT_DIR,
                polars_logic.read_parquet,
                polars_logic.write_parquet,
                getattr(polars_logic, f"{operation}_dataset"),
                operation,
                _method_name(operation),
                f"Write{_method_name(operation)}",
                other_dataset=other_dataset,
            )

def _method_name(operation: str) -> str:
    return {
        "filter": "Filter",
        "sort": "Sort",
        "pivot": "Pivot",
        "group_count": "GroupCount",
        "join": "Join",
    }[operation]


def main() -> None:
    BenchmarkRunner().run()


if __name__ == "__main__":
    main()
