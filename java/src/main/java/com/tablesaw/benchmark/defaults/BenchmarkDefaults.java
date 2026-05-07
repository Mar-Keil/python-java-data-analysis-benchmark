package com.tablesaw.benchmark.defaults;

import com.sun.management.OperatingSystemMXBean;
import com.tablesaw.logicTablesaw.TablesawLogic;
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
        "-XX:+UseG1GC",
        // Hadoop Reflection Workarounds für Java 16+ / 21 / 25
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    }
)
@State(Scope.Benchmark)
public abstract class BenchmarkDefaults {

  protected final TablesawLogic logic;
  protected final OperatingSystemMXBean os;

  private long realBefore;
  private long cpuBefore;

  private final Path dataOutDir;
  private final Path writeRootDir;

  @Param({"20k", "80k", "320k", "1280k", "5120k", "20480k"})
  protected String flightsDataset;

  protected BenchmarkDefaults() {
    this.logic = new TablesawLogic();
    this.os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

    Path cwd = Path.of("").toAbsolutePath().normalize();
    Path repoRoot = cwd.endsWith("java") ? cwd.getParent() : cwd;
    this.dataOutDir = repoRoot.resolve("data-gen/out");
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
