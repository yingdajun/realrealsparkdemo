package org.example.ml

// $example on$
import org.apache.spark.ml.fpm.PrefixSpan
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating PrefixSpan.
 * Run with
 * {{{
 * bin/run-example ml.PrefixSpanExample
 * }}}
 */
object PrefixSpanExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    val smallTestData = Seq(
      Seq(Seq(1, 2), Seq(3)),
      Seq(Seq(1), Seq(3, 2), Seq(1, 2)),
      Seq(Seq(1, 2), Seq(5)),
      Seq(Seq(6)))

    val df = smallTestData.toDF("sequence")
    val result = new PrefixSpan()
      .setMinSupport(0.5)
      .setMaxPatternLength(5)
      .setMaxLocalProjDBSize(32000000)
      .findFrequentSequentialPatterns(df)
      .show()
    // $example off$

    spark.stop()
  }
}