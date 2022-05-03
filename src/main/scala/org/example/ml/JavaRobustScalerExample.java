package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import org.apache.spark.ml.feature.RobustScaler;
import org.apache.spark.ml.feature.RobustScalerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$

public class JavaRobustScalerExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaRobustScalerExample")
                .getOrCreate();

        // $example on$
        Dataset<Row> dataFrame =
                spark.read().format("libsvm").load("D:\\sparkdata\\mllib\\sample_libsvm_data.txt");

        RobustScaler scaler = new RobustScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithScaling(true)
                .setWithCentering(false)
                .setLower(0.25)
                .setUpper(0.75);

        // Compute summary statistics by fitting the RobustScaler
        RobustScalerModel scalerModel = scaler.fit(dataFrame);

        // Transform each feature to have unit quantile range.
        Dataset<Row> scaledData = scalerModel.transform(dataFrame);
        scaledData.show();
        // $example off$
        spark.stop();
    }
}
