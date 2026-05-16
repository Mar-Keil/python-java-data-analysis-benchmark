import csv

from pathlib import Path

"""Provides the CSV writer print the benchmark results."""


class PrintCSV:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.engine_name = "polars"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self._create_csv()

    def _create_csv(self) -> Path:
        file_number = 1

        while True:
            candidate = self.output_dir / f"{file_number}_results.csv"
            if not candidate.exists():
                with candidate.open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ["engine", "benchmark_size", "method", "category", "score", "unit"]
                    )
                return candidate
            file_number += 1

    def write_result(
            self,
            benchmark_size: str,
            method: str,
            category: str,
            score: float,
            unit: str,
    ) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [self.engine_name, benchmark_size, method, category, score, unit]
            )

        print(
            f"[CSV][{self.engine_name}] "
            f"{benchmark_size} | {method} | {category} = {score} {unit}"
        )
