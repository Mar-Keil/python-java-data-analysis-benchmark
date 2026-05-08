package com.logic.optimized;

import org.dflib.DataFrame;

import com.logic.DFLibLogic;

public class OptimizedDFLibLogic {

  private final DFLibLogic logic = new DFLibLogic();
  private final Parallelizer parallelizer = new Parallelizer();

  public DataFrame filter(DataFrame flights) {
    return parallelizer.execute(flights, logic::filter);
  }

  public DataFrame join(DataFrame flights, DataFrame airlines) {
    return parallelizer.execute(flights, chunk -> logic.join(chunk, airlines));
  }
}
