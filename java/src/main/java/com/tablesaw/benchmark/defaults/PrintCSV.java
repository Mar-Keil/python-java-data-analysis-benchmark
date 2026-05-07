package com.tablesaw.benchmark.defaults;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
      String method = benchmarkName.substring(benchmarkName.lastIndexOf('.') + 1);

      double timeScore = result.getPrimaryResult().getScore();
      double cpuScore = result.getSecondaryResults().get("CPU").getScore();

      appendRow(benchmarkSize, method, "Time", timeScore, "s/op");
      appendRow(benchmarkSize, method, "CPU", cpuScore, "cores/op");
    }

    System.out.println("Benchmark CSV written to: " + csvPath.toAbsolutePath());
  }

  private void appendRow(String benchmarkSize, String method, String category, double score, String unit) {
        String row = String.join(
                ",",
                benchmarkSize,
                method,
                category,
                Double.toString(score),
                unit)
            + System.lineSeparator();

    try {
      Files.writeString(
          csvPath, row, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    } catch (IOException exception) {
        throw new IllegalStateException("Could not append benchmark result to CSV.", exception);
    }
  }
}
