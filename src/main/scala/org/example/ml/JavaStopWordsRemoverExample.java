package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

public class JavaStopWordsRemoverExample {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStopWordsRemoverExample")
                .getOrCreate();

        // $example on$
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("raw")
                .setOutputCol("filtered");

        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList("I", "saw", "the", "red", "balloon")),
                RowFactory.create(Arrays.asList("Mary", "had", "a", "little", "lamb"))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField(
                        "raw", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
        });

        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        remover.transform(dataset).show(false);
        // $example off$
        spark.stop();
    }
}
