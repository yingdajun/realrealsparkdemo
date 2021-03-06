package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.MinHashLSH;
import org.apache.spark.ml.feature.MinHashLSHModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;
// $example off$

/**
 * An example demonstrating MinHashLSH.
 * Run with:
 *   bin/run-example ml.JavaMinHashLSHExample
 */
public class JavaMinHashLSHExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMinHashLSHExample")
                .getOrCreate();

        // $example on$
        List<Row> dataA = Arrays.asList(
                RowFactory.create(0, Vectors.sparse(6, new int[]{0, 1, 2}, new double[]{1.0, 1.0, 1.0})),
                RowFactory.create(1, Vectors.sparse(6, new int[]{2, 3, 4}, new double[]{1.0, 1.0, 1.0})),
                RowFactory.create(2, Vectors.sparse(6, new int[]{0, 2, 4}, new double[]{1.0, 1.0, 1.0}))
        );

        List<Row> dataB = Arrays.asList(
                RowFactory.create(0, Vectors.sparse(6, new int[]{1, 3, 5}, new double[]{1.0, 1.0, 1.0})),
                RowFactory.create(1, Vectors.sparse(6, new int[]{2, 3, 5}, new double[]{1.0, 1.0, 1.0})),
                RowFactory.create(2, Vectors.sparse(6, new int[]{1, 2, 4}, new double[]{1.0, 1.0, 1.0}))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> dfA = spark.createDataFrame(dataA, schema);
        Dataset<Row> dfB = spark.createDataFrame(dataB, schema);

        int[] indices = {1, 3};
        double[] values = {1.0, 1.0};
        Vector key = Vectors.sparse(6, indices, values);

        MinHashLSH mh = new MinHashLSH()
                .setNumHashTables(5)
                .setInputCol("features")
                .setOutputCol("hashes");

        MinHashLSHModel model = mh.fit(dfA);

        // Feature Transformation
        System.out.println("The hashed dataset where hashed values are stored in the column 'hashes':");
        model.transform(dfA).show();

        // Compute the locality sensitive hashes for the input rows, then perform approximate
        // similarity join.
        // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
        // `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
        System.out.println("Approximately joining dfA and dfB on Jaccard distance smaller than 0.6:");
        model.approxSimilarityJoin(dfA, dfB, 0.6, "JaccardDistance")
                .select(col("datasetA.id").alias("idA"),
                        col("datasetB.id").alias("idB"),
                        col("JaccardDistance")).show();

        // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
        // neighbor search.
        // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
        // `model.approxNearestNeighbors(transformedA, key, 2)`
        // It may return less than 2 rows when not enough approximate near-neighbor candidates are
        // found.
        System.out.println("Approximately searching dfA for 2 nearest neighbors of the key:");
        model.approxNearestNeighbors(dfA, key, 2).show();
        // $example off$

        spark.stop();
    }
}
