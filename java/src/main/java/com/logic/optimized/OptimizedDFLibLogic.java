package com.logic.optimized;

import org.dflib.DataFrame;

import com.logic.DFLibLogic;

/**
 * Provides the entry points for the manually parallelized DFLib benchmark operations
 * and applies early exits for datasets that are too small to benefit from parallel execution.
 */
public class OptimizedDFLibLogic {

  private final int minRows = 125_000;

  private final DFLibLogic logic = new DFLibLogic();
  private final Parallelizer parallelizer = new Parallelizer(minRows);
  private final Sorter sorter = new Sorter();

  public boolean toSmall(int rows) {return rows <= minRows;}

  public DataFrame filter(DataFrame flights) {
    if (toSmall(flights.height())) return logic.filter(flights);
    return parallelizer.execute(flights, logic::filter);
  }

  public DataFrame join(DataFrame flights, DataFrame airlines) {
    if (toSmall(flights.height())) return logic.join(flights, airlines);
    return parallelizer.execute(flights, chunk -> logic.join(chunk, airlines));
  }

  public DataFrame pivot(DataFrame flights) {
    if (toSmall(flights.height())) return logic.pivot(flights);
    DataFrame partialResult = parallelizer.execute(flights, logic::pivot);
    return logic.mergePivot(partialResult);
  }

  public DataFrame groupCount(DataFrame flights) {
    if (toSmall(flights.height())) return logic.groupCount(flights);
    DataFrame partialResult = parallelizer.execute(flights, logic::groupCount);
    return logic.mergeGroupCount(partialResult);
  }

  public DataFrame sort(DataFrame flights) {
    if (toSmall(flights.height())) return logic.sort(flights);
    return sorter.execute(flights, minRows);
  }
}
