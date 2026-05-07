package com.benchmark.defaults;

import com.sun.management.OperatingSystemMXBean;
import com.logic.DFLibLogic;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 10, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 7, time = 10, timeUnit = TimeUnit.SECONDS)
@Fork(
    value = 1, 
    jvmArgsAppend = {
        "-Xms14G", 
        "-Xmx14G", 
        "-XX:+UseG1GC"
    }
)
@State(Scope.Benchmark)
public abstract class BenchmarkDefaults {

  protected final DFLibLogic logic;
  protected final OperatingSystemMXBean os;

  private long realBefore;
  private long cpuBefore;

  private final Path dataOutDir;
  private final Path writeRootDir;

  @Param({"31.25k", "125k", "500k", "2000k", "8000k"})
  protected String flightsDataset;

  protected BenchmarkDefaults() {
    this.logic = new DFLibLogic();
    this.os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

    Path cwd = Path.of("").toAbsolutePath().normalize();
    Path repoRoot = cwd.endsWith("java") ? cwd.getParent() : cwd;
    this.dataOutDir = repoRoot.resolve("data_gen/out");
    this.writeRootDir = repoRoot.resolve("java/out/jmh-write");
  }

  protected Path resolveFlightsPath(String datasetLabel) {
    return dataOutDir.resolve(datasetLabel + "Flights.parquet");
  }

  protected Path resolveAirlinesPath() {
    return dataOutDir.resolve("airlines.parquet");
  }

  protected Path resolveWriteOutputDir(String benchmarkName) {
    return writeRootDir.resolve(benchmarkName);
  }

  @Setup(Level.Iteration)
  public void setupIterationMetrics() {
    realBefore = System.nanoTime();
    cpuBefore = os.getProcessCpuTime();
  }

  @TearDown(Level.Iteration)
  public void tearDownIterationMetrics(ExtraMetrics metrics) {
    long realAfter = System.nanoTime();
    long cpuAfter = os.getProcessCpuTime();
    metrics.CPU = (double) (cpuAfter - cpuBefore) / (double) (realAfter - realBefore);
  }
}
