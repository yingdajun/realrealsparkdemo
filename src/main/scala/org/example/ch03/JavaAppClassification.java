package org.example.ch03;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class JavaAppClassification {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession
                .builder()
                .master("local[2]")
                .appName("JavaAppClassification")
                .getOrCreate();

        Dataset<Row> data = spark.read().format("libsvm")
                .load("D:\\sparkdata\\bigdata\\2rd_data\\ch03\\libsvm\\part-00000");
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

        // train the model
        NaiveBayesModel model = nb.fit(trainingData);

        // Select example rows to display.
        Dataset<Row> predictions = model.transform(testData);
        predictions.show();

        // compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy);

//        model.save("./demo");

        System.out.println("1");
        NaiveBayesModel sameModel = NaiveBayesModel.load("./demo");
        System.out.println(sameModel);
        System.out.println("2");


        spark.stop();

    }
}
