package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$

public class JavaStandardScalerExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStandardScalerExample")
                .getOrCreate();

        // $example on$
        Dataset<Row> dataFrame =
                spark.read().format("libsvm").load("D:\\sparkdata\\mllib\\sample_libsvm_data.txt");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(false);

        // Compute summary statistics by fitting the StandardScaler
        StandardScalerModel scalerModel = scaler.fit(dataFrame);

        // Normalize each feature to have unit standard deviation.
        Dataset<Row> scaledData = scalerModel.transform(dataFrame);
        scaledData.show();
        // $example off$
        spark.stop();
    }
}
