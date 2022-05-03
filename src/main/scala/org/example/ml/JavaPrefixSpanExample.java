package org.example.ml;


// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.fpm.PrefixSpan;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
// $example off$

/**
 * An example demonstrating PrefixSpan.
 * Run with
 * <pre>
 * bin/run-example ml.JavaPrefixSpanExample
 * </pre>
 */
public class JavaPrefixSpanExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPrefixSpanExample")
                .getOrCreate();

        // $example on$
        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList(Arrays.asList(1, 2), Arrays.asList(3))),
                RowFactory.create(Arrays.asList(Arrays.asList(1), Arrays.asList(3, 2), Arrays.asList(1,2))),
                RowFactory.create(Arrays.asList(Arrays.asList(1, 2), Arrays.asList(5))),
                RowFactory.create(Arrays.asList(Arrays.asList(6)))
        );
        StructType schema = new StructType(new StructField[]{ new StructField(
                "sequence", new ArrayType(new ArrayType(DataTypes.IntegerType, true), true),
                false, Metadata.empty())
        });
        Dataset<Row> sequenceDF = spark.createDataFrame(data, schema);

        PrefixSpan prefixSpan = new PrefixSpan().setMinSupport(0.5).setMaxPatternLength(5);

        // Finding frequent sequential patterns
        prefixSpan.findFrequentSequentialPatterns(sequenceDF).show();
        // $example off$

        spark.stop();
    }
}
