package com.tablesaw.benchmark.run;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.tablesaw.benchmark.defaults.BenchmarkDefaults;
import org.openjdk.jmh.annotations.*;
import tech.tablesaw.api.Table;

@State(Scope.Benchmark)
public class SortBenchmarks extends BenchmarkDefaults {

  private Table rawFlights;
  private Table sortedFlights;

  private Path output;
  
  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    sortedFlights = logic.sort(rawFlights);
    Path outputDir = resolveWriteOutputDir("sort");
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "Sort.parquet");
  }

  @Benchmark
  public Table sort() {
    sortedFlights = logic.sort(rawFlights);
    return sortedFlights;
  }

  @Benchmark
  public int writeSort() {
    logic.writeParquet(sortedFlights, output);
    return sortedFlights.rowCount();
  }
}
