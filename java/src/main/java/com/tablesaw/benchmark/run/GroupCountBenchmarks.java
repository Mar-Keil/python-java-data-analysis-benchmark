package com.tablesaw.benchmark.run;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.tablesaw.benchmark.defaults.BenchmarkDefaults;
import org.openjdk.jmh.annotations.*;
import tech.tablesaw.api.Table;

@State(Scope.Benchmark)
public class GroupCountBenchmarks extends BenchmarkDefaults {

  private Table rawFlights;
  private Table groupedFlights;
  private Path output;

  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    groupedFlights = logic.groupCount(rawFlights);
    Path outputDir = resolveWriteOutputDir("groupCount");
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "GroupCount.parquet");
  }

  @Benchmark
  public Table groupCount() {
    groupedFlights = logic.groupCount(rawFlights);
    return groupedFlights;
  }

  @Benchmark
  public int writeGroupCount() {
    logic.writeParquet(groupedFlights, output);
    return groupedFlights.rowCount();
  }
}
