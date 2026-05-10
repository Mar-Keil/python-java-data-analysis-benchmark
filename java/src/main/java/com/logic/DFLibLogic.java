package com.logic;

import static org.dflib.Exp.$int;
import static org.dflib.Exp.$str;
import static org.dflib.Exp.count;

import java.nio.file.Path;
import org.dflib.DataFrame;
import org.dflib.parquet.Parquet;

public class DFLibLogic {
  public DataFrame readParquet(Path input) {
    return Parquet.load(input);
  }

  public void writeParquet(DataFrame dataFrame, Path output) {
    Parquet.save(dataFrame, output);
  }

  public DataFrame filter(DataFrame flights) {
    return flights.rows($str("aircraft_model").eq("A319neo")).select();
  }

  public DataFrame filterRange(DataFrame flights, int lowerBound, int upperBound) {
    return flights.rows(
        $int("flight_number").ge(lowerBound).and($int("flight_number").lt(upperBound)))
        .select();
  }

  public DataFrame pivot(DataFrame flights) {
    return flights.group("aircraft_model").agg(
        $str("aircraft_model").first().as("aircraft_model"),
        $int("flight_distance").sum().as("sum_flight_distance"));
  }

  public DataFrame mergePivot(DataFrame groupedFlights) {
    return groupedFlights.group("aircraft_model").agg(
        $str("aircraft_model").first().as("aircraft_model"),
        $int("sum_flight_distance").sum().as("sum_flight_distance"));
  }

  public DataFrame groupCount(DataFrame flights) {
    return flights.group("aircraft_model", "airline_code").agg(
        $str("aircraft_model").first().as("aircraft_model"),
        $int("airline_code").first().as("airline_code"),
        count().as("count_aircraft"));
  }

  public DataFrame mergeGroupCount(DataFrame groupedFlights) {
    return groupedFlights.group("aircraft_model", "airline_code").agg(
        $str("aircraft_model").first().as("aircraft_model"),
        $int("airline_code").first().as("airline_code"),
        $int("count_aircraft").sum().as("count_aircraft"));
  }

  public DataFrame join(DataFrame flights, DataFrame airlines) {
    return flights.join(airlines).on("airline_code").colsExcept("airline_code_").select();
  }

  public DataFrame sort(DataFrame flights) {
    return flights.sort("flight_number", true);
  }
}
