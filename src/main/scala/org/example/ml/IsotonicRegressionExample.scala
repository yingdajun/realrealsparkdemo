package org.example.ml

// $example on$
import org.apache.spark.ml.regression.IsotonicRegression
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating Isotonic Regression.
 * Run with
 * {{{
 * bin/run-example ml.IsotonicRegressionExample
 * }}}
 */
object IsotonicRegressionExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()

    // $example on$
    // Loads data.
    val dataset = spark.read.format("libsvm")
      .load("D:\\sparkdata\\mllib\\sample_isotonic_regression_libsvm_data.txt")

    // Trains an isotonic regression model.
    val ir = new IsotonicRegression()
    val model = ir.fit(dataset)

    println(s"Boundaries in increasing order: ${model.boundaries}\n")
    println(s"Predictions associated with the boundaries: ${model.predictions}\n")

    // Makes predictions.
    model.transform(dataset).show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
