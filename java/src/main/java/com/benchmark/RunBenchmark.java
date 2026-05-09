package com.benchmark;

import com.benchmark.defaults.PrintCSV;
import com.benchmark.run.OperationBenchmarks;
import com.benchmark.run.ReadBenchmarks;
import com.benchmark.run.WriteBenchmarks;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public final class RunBenchmark {

  public static void main(String[] args) throws RunnerException {

    PrintCSV printCSV = new PrintCSV();

    Options options =
        new OptionsBuilder()
            .include(ReadBenchmarks.class.getName())
            .include(OperationBenchmarks.class.getName())
            .include(WriteBenchmarks.class.getName())
            .build();

    printCSV.runAndWrite(new Runner(options).run());
  }
}
