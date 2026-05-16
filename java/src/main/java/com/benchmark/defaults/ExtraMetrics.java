package com.benchmark.defaults;

import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

/**
 * Stores the extra CPU metric.
 */

@AuxCounters(AuxCounters.Type.EVENTS)
@State(Scope.Thread)
public class ExtraMetrics {
  public double CPU = 0;
}
