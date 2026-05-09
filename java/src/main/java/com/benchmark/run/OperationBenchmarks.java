package com.benchmark.run;

import com.benchmark.defaults.BenchmarkDefaults;
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

  @Setup(Level.Trial)
  public void setupTrial(){
    rawFlights = logic.readParquet(resolveFlightsPath(flightsDataset));
    rawAirlines = logic.readParquet(resolveAirlinesPath());
  }

  @Benchmark
  public DataFrame runOperation() {
    return switch (operation) {
      case "filter" -> logic.filter(rawFlights);
      case "sort" -> logic.sort(rawFlights);
      case "pivot" -> logic.pivot(rawFlights);
      case "groupCount" -> logic.groupCount(rawFlights);
      case "join" -> logic.join(rawFlights, rawAirlines);
      default -> throw new IllegalStateException("Unsupported operation: " + operation);
    };
  }

  @Benchmark
  public DataFrame runOptimizedOperation() {
    return switch (operation) {
      case "filter" -> optimizedLogic.filter(rawFlights);
      case "sort" -> optimizedLogic.sort(rawFlights);
      case "pivot" -> optimizedLogic.pivot(rawFlights);
      case "groupCount" -> optimizedLogic.groupCount(rawFlights);
      case "join" -> optimizedLogic.join(rawFlights, rawAirlines);
      default -> throw new IllegalStateException("Unsupported operation: " + operation);
    };
  }
}
