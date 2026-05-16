package com.benchmark.run;

import com.benchmark.defaults.BenchmarkDefaults;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.dflib.DataFrame;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

/**
 * Benchmarks writing the sorted flight dataset to a parquet file with DFLib.
 */
@State(Scope.Benchmark)
public class WriteBenchmarks extends BenchmarkDefaults {

  private DataFrame sortedFlights;
  private Path output;

  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    DataFrame rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    sortedFlights = logic.sort(rawFlights);

    Path outputDir = resolveWriteSortOutputDir();
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "Sort.parquet");
  }

  @Benchmark
  public int writeSortResult() {
    logic.writeParquet(sortedFlights, output);
    return sortedFlights.height();
  }
}
