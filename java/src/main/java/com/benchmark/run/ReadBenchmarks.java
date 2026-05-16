package com.benchmark.run;

import com.benchmark.defaults.BenchmarkDefaults;
import org.dflib.DataFrame;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

/**
 * Benchmarks reading the flight and the airlines parquet datasets with DFLib.
 */
@State(Scope.Benchmark)
public class ReadBenchmarks extends BenchmarkDefaults {

  @Benchmark
  public int readDatasets() {
    DataFrame flights = logic.readParquet(resolveFlightsPath(flightsDataset));
    DataFrame airlines = logic.readParquet(resolveAirlinesPath());
    return flights.height() + airlines.height();
  }
}
