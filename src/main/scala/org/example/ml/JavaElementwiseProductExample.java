package org.example.ml;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

// $example on$
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.ElementwiseProduct;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off$

//元素智能乘积-ElementwiseProduct
public class JavaElementwiseProductExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaElementwiseProductExample")
                .getOrCreate();

        // $example on$
        // Create some vector data; also works for sparse vectors
        List<Row> data = Arrays.asList(
                RowFactory.create("a", Vectors.dense(1.0, 2.0, 3.0)),
                RowFactory.create("b", Vectors.dense(4.0, 5.0, 6.0))
        );

        List<StructField> fields = new ArrayList<>(2);
        fields.add(DataTypes.createStructField("id", DataTypes.StringType, false));
        fields.add(DataTypes.createStructField("vector", new VectorUDT(), false));

        StructType schema = DataTypes.createStructType(fields);

        Dataset<Row> dataFrame = spark.createDataFrame(data, schema);

        Vector transformingVector = Vectors.dense(0.0, 1.0, 2.0);

        ElementwiseProduct transformer = new ElementwiseProduct()
                .setScalingVec(transformingVector)
                .setInputCol("vector")
                .setOutputCol("transformedVector");

        // Batch transform the vectors to create new column:
        transformer.transform(dataFrame).show();
        // $example off$
        spark.stop();
    }
}
