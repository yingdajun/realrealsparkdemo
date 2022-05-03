package org.example.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
// $example off$
import org.apache.spark.sql.SparkSession

object FMRegressorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("FMRegressorExample")
      .getOrCreate()

    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("D:\\sparkdata\\mllib\\sample_libsvm_data.txt")

    // Scale features.
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a FM model.
    val fm = new FMRegressor()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)

    // Create a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureScaler, fm))

    // Train model.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val fmModel = model.stages(1).asInstanceOf[FMRegressionModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
