package com.logic.optimized;

import org.dflib.DataFrame;

import com.logic.DFLibLogic;

public class OptimizedDFLibLogic {

  private final int min_rows = 125_000;

  private final DFLibLogic logic = new DFLibLogic();
  private final Parallelizer parallelizer = new Parallelizer(min_rows);

  public boolean toSmall(int rows) {return rows <= min_rows;}

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
    return logic.pivot(partialResult);
  }

  public DataFrame groupCount(DataFrame flights) {
    if (toSmall(flights.height())) return logic.groupCount(flights);
    DataFrame partialResult = parallelizer.execute(flights, logic::groupCount);
    return logic.mergeGroupCount(partialResult);
  }
}
