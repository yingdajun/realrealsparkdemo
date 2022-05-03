package org.example.ml;


import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

public class JavaMinMaxScalerExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMinMaxScalerExample")
                .getOrCreate();

        // $example on$
        List<Row> data = Arrays.asList(
                RowFactory.create(0, Vectors.dense(1.0, 0.1, -1.0)),
                RowFactory.create(1, Vectors.dense(2.0, 1.1, 1.0)),
                RowFactory.create(2, Vectors.dense(3.0, 10.1, 3.0))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> dataFrame = spark.createDataFrame(data, schema);

        MinMaxScaler scaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");

        // Compute summary statistics and generate MinMaxScalerModel
        MinMaxScalerModel scalerModel = scaler.fit(dataFrame);

        // rescale each feature to range [min, max].
        Dataset<Row> scaledData = scalerModel.transform(dataFrame);
        System.out.println("Features scaled to range: [" + scaler.getMin() + ", "
                + scaler.getMax() + "]");
        scaledData.select("features", "scaledFeatures").show();
        // $example off$

        spark.stop();
    }
}
