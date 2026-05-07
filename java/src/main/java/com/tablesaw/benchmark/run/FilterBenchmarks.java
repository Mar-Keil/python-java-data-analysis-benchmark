package com.tablesaw.benchmark.run;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.tablesaw.benchmark.defaults.BenchmarkDefaults;
import org.openjdk.jmh.annotations.*;
import tech.tablesaw.api.Table;

@State(Scope.Benchmark)
public class FilterBenchmarks extends BenchmarkDefaults {

  private Table rawFlights;
  private Table filteredFlights;
  private Path output;

  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    filteredFlights = logic.filter(rawFlights);
    Path outputDir = resolveWriteOutputDir("filter");
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "Filter.parquet");
  }

  @Benchmark
  public Table filter() {
    filteredFlights = logic.filter(rawFlights);
    return filteredFlights;
  }

  @Benchmark
  public int writeFilter() {
    logic.writeParquet(filteredFlights, output);
    return filteredFlights.rowCount(); // Verhindert Dead-Code-Elimination
  }
}
