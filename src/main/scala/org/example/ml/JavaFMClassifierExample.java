package org.example.ml;

// $example on$
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.FMClassificationModel;
import org.apache.spark.ml.classification.FMClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
// $example off$

public class JavaFMClassifierExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaFMClassifierExample")
                .getOrCreate();

        // $example on$
        // Load and parse the data file, converting it to a DataFrame.
        Dataset<Row> data = spark
                .read()
                .format("libsvm")
                .load("D:\\sparkdata\\mllib\\sample_libsvm_data.txt");

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);
        // Scale features.
        MinMaxScalerModel featureScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .fit(data);

        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a FM model.
        FMClassifier fm = new FMClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("scaledFeatures")
                .setStepSize(0.001);

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        // Create a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureScaler, fm, labelConverter});

        // Train model.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5);

        // Select (prediction, true label) and compute test accuracy.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Accuracy = " + accuracy);

        FMClassificationModel fmModel = (FMClassificationModel)(model.stages()[2]);
        System.out.println("Factors: " + fmModel.factors());
        System.out.println("Linear: " + fmModel.linear());
        System.out.println("Intercept: " + fmModel.intercept());
        // $example off$

        spark.stop();
    }
}
