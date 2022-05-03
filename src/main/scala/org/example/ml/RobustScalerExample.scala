package org.example.ml

// $example on$
import org.apache.spark.ml.feature.RobustScaler
// $example off$
import org.apache.spark.sql.SparkSession

object RobustScalerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RobustScalerExample")
      .getOrCreate()

    // $example on$
    val dataFrame = spark.read.format("libsvm").load("D:\\sparkdata\\mllib\\sample_libsvm_data.txt")

    val scaler = new RobustScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithScaling(true)
      .setWithCentering(false)
      .setLower(0.25)
      .setUpper(0.75)

    // Compute summary statistics by fitting the RobustScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Transform each feature to have unit quantile range.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
