package com.benchmark.run;

import com.benchmark.defaults.BenchmarkDefaults;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.dflib.DataFrame;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class OperationBenchmarks extends BenchmarkDefaults {

  @Param({"filter", "sort", "pivot", "groupCount", "join"})
  private String operation;

  private DataFrame rawFlights;
  private DataFrame rawAirlines;
  private DataFrame operationResult;
  private Path output;

  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    rawAirlines = logic.readParquet(resolveAirlinesPath());
    operationResult = runSelectedOperation();

    Path outputDir = resolveWriteOutputDir(operation);
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + capitalize(operation) + ".parquet");
  }

  @Benchmark
  public DataFrame runOperation() {
    operationResult = runSelectedOperation();
    return operationResult;
  }

  @Benchmark
  public int writeOperation() {
    logic.writeParquet(operationResult, output);
    return operationResult.height();
  }

  private DataFrame runSelectedOperation() {
    return switch (operation) {
      case "filter" -> logic.filter(rawFlights);
      case "sort" -> logic.sort(rawFlights);
      case "pivot" -> logic.pivot(rawFlights);
      case "groupCount" -> logic.groupCount(rawFlights);
      case "join" -> logic.join(rawFlights, rawAirlines);
      default -> throw new IllegalStateException("Unsupported operation: " + operation);
    };
  }

  private String capitalize(String value) {
    return value.substring(0, 1).toUpperCase() + value.substring(1);
  }
}
