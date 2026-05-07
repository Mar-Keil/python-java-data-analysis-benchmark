package com.tablesaw.benchmark;

import com.tablesaw.benchmark.defaults.PrintCSV;
import com.tablesaw.benchmark.run.FilterBenchmarks;
import com.tablesaw.benchmark.run.GroupCountBenchmarks;
import com.tablesaw.benchmark.run.JoinBenchmarks;
import com.tablesaw.benchmark.run.PivotBenchmarks;
import com.tablesaw.benchmark.run.ReadBenchmarks;
import com.tablesaw.benchmark.run.SortBenchmarks;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public final class RunBenchmark {

  public static void main(String[] args) throws RunnerException {

    PrintCSV printCSV = new PrintCSV();

    Options options =
        new OptionsBuilder()
            /*
            .include(ReadBenchmarks.class.getName())
            .include(FilterBenchmarks.class.getName())
            .include(GroupCountBenchmarks.class.getName())
            .include(PivotBenchmarks.class.getName())
            .include(SortBenchmarks.class.getName())
            */
            .include(JoinBenchmarks.class.getName())
            .build();

    printCSV.runAndWrite(new Runner(options).run());
  }
}
