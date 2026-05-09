package com.logic.optimized;

import com.logic.DFLibLogic;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.dflib.DataFrame;
import org.dflib.Series;

public class Sorter {
  private final DFLibLogic logic = new DFLibLogic();

  public DataFrame execute(DataFrame flights, int minRows) {
    int chunkCount = Math.clamp(
        Math.ceilDiv(flights.height(), minRows),
        1,
        Runtime.getRuntime().availableProcessors());

    int[] minMax = getSortRange(flights.<Integer>getColumn("flight_number"));

    int rangeWidth = Math.ceilDiv(minMax[1] - minMax[0], chunkCount);

    try (ExecutorService executor = Executors.newFixedThreadPool(chunkCount)) {
      List<Future<DataFrame>> futures = new ArrayList<>(chunkCount);

      for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
        int lowerBound = minMax[0] + chunkIndex * rangeWidth;
        int upperBound = Math.min(lowerBound + rangeWidth, minMax[1]);

        futures.add(executor.submit(() -> {
          DataFrame range = logic.filterRange(flights, lowerBound, upperBound);
          return logic.sort(range);
        }));
      }
      return new Parallelizer(minRows).mergeResults(futures);
    }
  }

  private int[] getSortRange(Series<Integer> flightNumbers) {
    int min = flightNumbers.get(0);
    int max = flightNumbers.get(0);

    for (int i = 1; i < flightNumbers.size(); i++) {
      int current = flightNumbers.get(i);
      if (current < min) min = current;
      if (current > max) max = current;
    }

    return new int[] {min, max + 1};
  }
}
