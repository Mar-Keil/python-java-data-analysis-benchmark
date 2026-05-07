package com.tablesaw.benchmark.run;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.tablesaw.benchmark.defaults.BenchmarkDefaults;
import org.openjdk.jmh.annotations.*;
import tech.tablesaw.api.Table;

@State(Scope.Benchmark)
public class JoinBenchmarks extends BenchmarkDefaults {

  private Table rawFlights;
  private Table rawAirlines;
  private Table joinedFlights; 

  private Path output;
  
  @Setup(Level.Trial)
  public void setupTrial() throws IOException {
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    rawAirlines = logic.readParquet(resolveAirlinesPath());
    
    joinedFlights = logic.join(rawFlights, rawAirlines);
    
    Path outputDir = resolveWriteOutputDir("join");
    Files.createDirectories(outputDir);
    output = outputDir.resolve(flightsDataset + "Join.parquet");
  }

  @Benchmark
  public Table join() {
    return logic.join(rawFlights, rawAirlines);
  }

  @Benchmark
  public int writeJoin() {
    logic.writeParquet(joinedFlights, output);
    return joinedFlights.rowCount();
  }
}
