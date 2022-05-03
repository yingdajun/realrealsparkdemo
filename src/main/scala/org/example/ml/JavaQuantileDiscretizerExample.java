package org.example.ml;

import org.apache.spark.sql.SparkSession;
// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.QuantileDiscretizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

public class JavaQuantileDiscretizerExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaQuantileDiscretizerExample")
                .getOrCreate();

        // $example on$
        List<Row> data = Arrays.asList(
                RowFactory.create(0, 18.0),
                RowFactory.create(1, 19.0),
                RowFactory.create(2, 8.0),
                RowFactory.create(3, 5.0),
                RowFactory.create(4, 2.2)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("hour", DataTypes.DoubleType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);
        // $example off$
        // Output of QuantileDiscretizer for such small datasets can depend on the number of
        // partitions. Here we force a single partition to ensure consistent results.
        // Note this is not necessary for normal use cases
        df = df.repartition(1);
        // $example on$
        QuantileDiscretizer discretizer = new QuantileDiscretizer()
                .setInputCol("hour")
                .setOutputCol("result")
                .setNumBuckets(3);

        Dataset<Row> result = discretizer.fit(df).transform(df);
        result.show(false);
        // $example off$
        spark.stop();
    }
}