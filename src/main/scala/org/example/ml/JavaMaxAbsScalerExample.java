package org.example.ml;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
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
import org.apache.spark.sql.SparkSession;

public class JavaMaxAbsScalerExample {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMaxAbsScalerExample")
                .getOrCreate();

        // $example on$
        List<Row> data = Arrays.asList(
                RowFactory.create(0, Vectors.dense(1.0, 0.1, -8.0)),
                RowFactory.create(1, Vectors.dense(2.0, 1.0, -4.0)),
                RowFactory.create(2, Vectors.dense(4.0, 10.0, 8.0))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> dataFrame = spark.createDataFrame(data, schema);

        MaxAbsScaler scaler = new MaxAbsScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");

        // Compute summary statistics and generate MaxAbsScalerModel
        MaxAbsScalerModel scalerModel = scaler.fit(dataFrame);

        // rescale each feature to range [-1, 1].
        Dataset<Row> scaledData = scalerModel.transform(dataFrame);
        scaledData.select("features", "scaledFeatures").show();
        // $example off$

        spark.stop();
    }

}