package org.example.ch06


import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by seawalker on 2016/11/19.
 * 用户标识ID 电影标识ID集合
 * 136547	93,136,21,73,76
 */
object MoviesFP {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    //spark-warehouse 2rd_data/ch06/ar/part-00000 output/ch06/model local[2]
//    val Array(whdir,input,output,mode) = args

    val input="D:\\sparkdata\\bigdata\\2rd_data\\ch06\\ar\\part-00000"
    val output="./output/ch06/model"
    val mode="local[2]"
    val conf = new SparkConf()
      .setAppName(this.getClass.getSimpleName)
//      .set("spark.sql.warehouse.dir",whdir)
      .setMaster(mode)
    val sc = new SparkContext(conf)

    val transactions = sc.textFile(input).map(_.split("\t")).map(_(1).split(","))
    val minSupport = 0.005
    val minConfidence = 0.1

    val fpg = new FPGrowth().setMinSupport(minSupport).setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach
    { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    model.generateAssociationRules(minConfidence).collect().foreach
    { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]") + " => "
          + rule.consequent.mkString("[", ",", "]") + ", "
          + rule.confidence)
    }

    model.save(sc, output)
    sc.stop()
  }
}
