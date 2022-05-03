package org.example.ml;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.clustering.PowerIterationClustering;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

public class JavaPowerIterationClusteringExample {
    public static void main(String[] args) {
        // Create a SparkSession.
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPowerIterationClustering")
                .getOrCreate();

        // $example on$
        List<Row> data = Arrays.asList(
                RowFactory.create(0L, 1L, 1.0),
                RowFactory.create(0L, 2L, 1.0),
                RowFactory.create(1L, 2L, 1.0),
                RowFactory.create(3L, 4L, 1.0),
                RowFactory.create(4L, 0L, 0.1)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("src", DataTypes.LongType, false, Metadata.empty()),
                new StructField("dst", DataTypes.LongType, false, Metadata.empty()),
                new StructField("weight", DataTypes.DoubleType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);

        PowerIterationClustering model = new PowerIterationClustering()
                .setK(2)
                .setMaxIter(10)
                .setInitMode("degree")
                .setWeightCol("weight");

        Dataset<Row> result = model.assignClusters(df);
        result.show(false);
        // $example off$
        spark.stop();
    }
}
