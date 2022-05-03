package org.example.ml;

import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.RFormula;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.*;
// $example off$

public class JavaRFormulaExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaRFormulaExample")
                .getOrCreate();

        // $example on$
        StructType schema = createStructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("country", StringType, false),
                createStructField("hour", IntegerType, false),
                createStructField("clicked", DoubleType, false)
        });

        List<Row> data = Arrays.asList(
                RowFactory.create(7, "US", 18, 1.0),
                RowFactory.create(8, "CA", 12, 0.0),
                RowFactory.create(9, "NZ", 15, 0.0)
        );

        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        RFormula formula = new RFormula()
                .setFormula("clicked ~ country + hour")
                .setFeaturesCol("features")
                .setLabelCol("label");
        Dataset<Row> output = formula.fit(dataset).transform(dataset);
        output.select("features", "label").show();
        // $example off$
        spark.stop();
    }
}
