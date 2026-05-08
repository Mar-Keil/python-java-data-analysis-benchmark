package com.logic.optimized;

import java.util.ArrayList;
import java.util.function.Function;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.dflib.DataFrame;

public class Parallelizer {
  private static final int MIN_ROWS_PER_CHUNK = 125_000;

  public List<ChunkRange> split(int rowCount) {
    int chunkCount = Math.clamp(
        Math.ceilDiv(rowCount, MIN_ROWS_PER_CHUNK),
        1,
        Runtime.getRuntime().availableProcessors());

    int chunkSize = Math.ceilDiv(rowCount, chunkCount);

    List<ChunkRange> ranges = new ArrayList<>(chunkCount);

    for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
      int start = chunkIndex * chunkSize;
      int end = Math.min(start + chunkSize, rowCount);
      ranges.add(new ChunkRange(start, end));
    }

    return ranges;
  }

  public DataFrame execute(DataFrame flights, Function<DataFrame, DataFrame> operation) {
    List<ChunkRange> chunks = split(flights.height());

    if (chunks.size() == 1) {
      return operation.apply(flights);
    }

    try (ExecutorService executor = Executors.newFixedThreadPool(chunks.size())) {
      List<Future<DataFrame>> futures = new ArrayList<>(chunks.size());
      for (ChunkRange chunkRange : chunks) {
        futures.add(executor.submit(() -> operation.apply(
            flights.rowsRange(chunkRange.startInclusive(), chunkRange.endExclusive()).select())));
      }

      return mergeResults(futures);
    }
  }

  public DataFrame mergeResults(List<Future<DataFrame>> futures) {
    List<DataFrame> chunks = new ArrayList<>(futures.size());

    for (Future<DataFrame> future : futures) {
      try {
        chunks.add(future.get());
      } catch (InterruptedException | ExecutionException e) {
        if (e instanceof InterruptedException) {
          Thread.currentThread().interrupt();
        }
        throw new IllegalStateException("Failed to collect chunk results", e);
      }
    }

    return chunks.getFirst().vConcat(chunks.subList(1, chunks.size()).toArray(DataFrame[]::new));
  }
}
