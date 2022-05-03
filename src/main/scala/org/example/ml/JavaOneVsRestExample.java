package org.example.ml;

// $example on$
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$
import org.apache.spark.sql.SparkSession;


/**
 * An example of Multiclass to Binary Reduction with One Vs Rest,
 * using Logistic Regression as the base classifier.
 * Run with
 * <pre>
 * bin/run-example ml.JavaOneVsRestExample
 * </pre>
 */
public class JavaOneVsRestExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaOneVsRestExample")
                .getOrCreate();

        // $example on$
        // load data file.
        Dataset<Row> inputData = spark.read().format("libsvm")
                .load("D:\\sparkdata\\mllib\\sample_multiclass_classification_data.txt");

        // generate the train/test split.
        Dataset<Row>[] tmp = inputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> train = tmp[0];
        Dataset<Row> test = tmp[1];

        // configure the base classifier.
        LogisticRegression classifier = new LogisticRegression()
                .setMaxIter(10)
                .setTol(1E-6)
                .setFitIntercept(true);

        // instantiate the One Vs Rest Classifier.
        OneVsRest ovr = new OneVsRest().setClassifier(classifier);

        // train the multiclass model.
        OneVsRestModel ovrModel = ovr.fit(train);

        // score the model on test data.
        Dataset<Row> predictions = ovrModel.transform(test)
                .select("prediction", "label");

        // obtain evaluator.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");

        // compute the classification error on test data.
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1 - accuracy));
        // $example off$

        spark.stop();
    }

}
