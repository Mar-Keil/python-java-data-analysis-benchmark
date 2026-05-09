package com.benchmark.defaults;

import static com.benchmark.defaults.BenchmarkConfig.MEASUREMENT_ITERATIONS;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Collection;
import org.openjdk.jmh.results.RunResult;

public class PrintCSV {

  private final Path outputDir;
  private final Path csvPath;

  public PrintCSV() {
    Path cwd = Path.of("").toAbsolutePath().normalize();
    Path repoRoot = cwd.endsWith("java") ? cwd.getParent() : cwd;
    this.outputDir = repoRoot.resolve("java/out/csv");
    this.csvPath = createCsv();
  }

  private Path createCsv() {
    try {
      Files.createDirectories(outputDir);

      Path candidate;
      int fileNumber = 1;

      do {
        candidate = outputDir.resolve(fileNumber + "_results.csv");
        fileNumber++;
      } while (Files.exists(candidate));

      Files.writeString(
          candidate,
          "benchmark_size,method,category,score,unit" + System.lineSeparator(),
          StandardCharsets.UTF_8,
          StandardOpenOption.CREATE_NEW);

      return candidate;
    } catch (IOException exception) {
      throw new IllegalStateException("Could not create CSV output file.", exception);
    }
  }

  public void runAndWrite(Collection<RunResult> results) {
    for (RunResult result : results) {
      String benchmarkSize = result.getParams().getParam("flightsDataset");

      String benchmarkName = result.getParams().getBenchmark();
      String benchmarkMethod = benchmarkName.substring(benchmarkName.lastIndexOf('.') + 1);
      String operation = resolveOperationParam(result);
      String method = resolveMethodName(operation, benchmarkMethod);

      double timeScore = result.getPrimaryResult().getScore();
      double cpuScore =
          result.getSecondaryResults().get("CPU").getScore()
              / MEASUREMENT_ITERATIONS;

      appendRow(benchmarkSize, method, "Time", timeScore, "s/op");
      appendRow(benchmarkSize, method, "CPU", cpuScore, "cores/op");
    }

    System.out.println("Benchmark CSV written to: " + csvPath.toAbsolutePath());
  }

  private String resolveOperationParam(RunResult result) {
    try {
      return result.getParams().getParam("operation");
    } catch (IllegalArgumentException exception) {
      return null;
    }
  }

  private String resolveMethodName(String operation, String benchmarkMethod) {
    if (operation == null) {
      return benchmarkMethod;
    }

    String capitalizedOperation = operation.substring(0, 1).toUpperCase() + operation.substring(1);

    if ("runOperation".equals(benchmarkMethod)) {
      return capitalizedOperation;
    }
    if ("writeSortResult".equals(benchmarkMethod)) {
      return "WriteSort";
    }

    return benchmarkMethod;
  }

  private void appendRow(String benchmarkSize, String method, String category, double score, String unit) {
        String row = String.join(
                ",",
                benchmarkSize,
                method,
                category,
                formatScore(score),
                unit)
            + System.lineSeparator();

    try {
      Files.writeString(
          csvPath, row, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    } catch (IOException exception) {
        throw new IllegalStateException("Could not append benchmark result to CSV.", exception);
    }
  }

  private String formatScore(double score) {
    return String.format(Locale.ROOT, "%.12f", score);
  }
}
