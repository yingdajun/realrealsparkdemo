package org.example.ml

// $example on$
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
// $example off$

object TokenizerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("TokenizerExample")
      .getOrCreate()

    // $example on$
    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens = udf { (words: Seq[String]) => words.length }

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
