from pathlib import Path
from typing import Any
from typing import Callable

from benchmark.default.default_values import BENCHMARK_ITERATIONS
from benchmark.default.default_values import BENCHMARK_WARMUP_ITERATIONS
from benchmark.default.default_values import PARAM
from benchmark.default.invocation_loop import InvocationLoop
from benchmark.default.print_csv import PrintCSV
from benchmark.default.time_cpu_measurement import TimeCPUMeasurement
from benchmark.run.benchmark_operation import clear_result_directory
from benchmark.run.benchmark_operation import prepare_result_directory
from logic import polars_logic

"""Provides the benchmark routine for writing the generated operation results."""


def benchmark_write(
    print_csv: PrintCSV,
    output_dir: Path,
    read_function: Callable[[Path], Any],
    write_function: Callable[[Any, Path], None],
) -> None:
    write_measurement = TimeCPUMeasurement(print_csv)

    for path in PARAM:
        dataset = read_function(path)
        operation_result = polars_logic.sort_dataset(dataset)
        result_path = output_dir / "sort"
        prepare_result_directory(result_path)

        for iteration_index in range(BENCHMARK_ITERATIONS):
            should_measure = iteration_index >= BENCHMARK_WARMUP_ITERATIONS
            invocation_loop_write = InvocationLoop()
            invocation_loop_write.start()
            write_invocation = 0

            while invocation_loop_write.get_is_looping():
                write_invocation += 1
                write_output_path = (
                    result_path / f"{write_invocation}_sort.parquet"
                )

                if should_measure:
                    write_measurement.start()

                write_function(operation_result, write_output_path)

                if should_measure:
                    write_measurement.stop()

            invocation_loop_write.cancel()

            is_last_iteration = iteration_index == BENCHMARK_ITERATIONS - 1
            if not is_last_iteration:
                clear_result_directory(result_path)

        benchmark_size = path.stem.replace("Flights", "")
        write_measurement.write_results(benchmark_size, "WriteSort")
