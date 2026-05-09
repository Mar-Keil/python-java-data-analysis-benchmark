import shutil

from pathlib import Path
from typing import Any
from typing import Callable

from benchmark.default.default_values import BENCHMARK_ITERATIONS
from benchmark.default.default_values import BENCHMARK_WARMUP_ITERATIONS
from benchmark.default.default_values import PARAM
from benchmark.default.invocation_loop import InvocationLoop
from benchmark.default.print_csv import PrintCSV
from benchmark.default.time_cpu_measurement import TimeCPUMeasurement


def prepare_result_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_result_directory(path: Path) -> None:
    if not path.exists():
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def benchmark_operation(
    print_csv: PrintCSV,
    read_function: Callable[[Path], Any],
    operation_function: Callable[..., Any],
    method_name: str,
    other_dataset: Any | None = None,
) -> None:
    logic_measurement = TimeCPUMeasurement(print_csv)

    for path in PARAM:
        dataset = read_function(path)

        for iteration_index in range(BENCHMARK_ITERATIONS):
            should_measure = iteration_index >= BENCHMARK_WARMUP_ITERATIONS
            invocation_loop_logic = InvocationLoop()
            invocation_loop_logic.start()

            while invocation_loop_logic.get_is_looping():
                if should_measure:
                    logic_measurement.start()

                if other_dataset is None:
                    operation_result = operation_function(dataset)
                else:
                    operation_result = operation_function(dataset, other_dataset)

                if should_measure:
                    logic_measurement.stop()

                del operation_result

            invocation_loop_logic.cancel()

        benchmark_size = path.stem.replace("Flights", "")
        logic_measurement.write_results(benchmark_size, method_name)
