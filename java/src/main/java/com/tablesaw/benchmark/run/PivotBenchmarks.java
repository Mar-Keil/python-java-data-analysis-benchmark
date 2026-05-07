package com.tablesaw.benchmark.run;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.tablesaw.benchmark.defaults.BenchmarkDefaults;
import org.openjdk.jmh.annotations.*;
import tech.tablesaw.api.Table;

@State(Scope.Benchmark)
public class PivotBenchmarks extends BenchmarkDefaults {

  private Table rawFlights;
  private Table pivotedFlights;

  private Path output;
  
  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    pivotedFlights = logic.pivot(rawFlights);
    Path outputDir = resolveWriteOutputDir("pivot");
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "Pivot.parquet");
  }

  @Benchmark
  public Table pivot() {
    pivotedFlights = logic.pivot(rawFlights);
    return pivotedFlights;
  }

  @Benchmark
  public int writePivot() {
    logic.writeParquet(pivotedFlights, output);
    return pivotedFlights.rowCount();
  }

}
