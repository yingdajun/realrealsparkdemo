package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

public class JavaPolynomialExpansionExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPolynomialExpansionExample")
                .getOrCreate();

        // $example on$
        PolynomialExpansion polyExpansion = new PolynomialExpansion()
                .setInputCol("features")
                .setOutputCol("polyFeatures")
                .setDegree(3);

        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.dense(2.0, 1.0)),
                RowFactory.create(Vectors.dense(0.0, 0.0)),
                RowFactory.create(Vectors.dense(3.0, -1.0))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });
        Dataset<Row> df = spark.createDataFrame(data, schema);

        Dataset<Row> polyDF = polyExpansion.transform(df);
        polyDF.show(false);
        // $example off$

        spark.stop();
    }
}
