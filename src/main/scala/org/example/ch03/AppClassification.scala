package org.example.ch03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

/**
 * whdir: omitted.
 * input: libsvm file.
 * output: model store path.
 * mode: yarn-client, yarn-cluster or local[*].
 * spark-warehouse 2rd_data/ch03/libsvm/part-00000 output/ch03/model local[2]
 */

object AppClassification {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    //    val Array(whdir, input, output, mode) = args
    //    val spark = SparkSession
    //      .builder
    ////      .config("spark.sql.warehouse.dir", whdir)
    //      .master(mode)
    //      .appName(this.getClass.getName)
    //      .getOrCreate()

    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("AppClassification")
      .getOrCreate()

    val data = spark.read.format("libsvm").load("D:\\sparkdata\\bigdata\\2rd_data\\ch03\\libsvm\\part-00000")
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // Train a NaiveBayes model.
    val model = new NaiveBayes().fit(trainingData)
    val predictions = model.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)

    /**
     * |-- label: double (nullable = true)
     * |-- features: vector (nullable = true)
     * |-- rawPrediction: vector (nullable = true)
     * |-- probability: vector (nullable = true)
     * |-- prediction: double (nullable = true)
     */
    model.save("./result1")

    spark.stop()
  }
}
