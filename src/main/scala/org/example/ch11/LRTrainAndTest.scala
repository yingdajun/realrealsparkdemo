package org.example.ch11
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object LRTrainAndTest {

  def main(args: Array[String]) {

//    if (args.length < 8) {
//      System.err.println("Usage:LRTrainAndTest <trainingPath> <testPath> <output> <numFeatures> <partitions> <RegParam> <NumIterations> <NumCorrections>")
//      System.exit(1)
//    }

    //2rd_data/ch11/test/part-00000 2rd_data/ch11/training/part-00000 output/ch11/label 23937 50 0.01 100 10
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ADTest with logistic regression")
    val sc = new SparkContext(conf)

    val arg1="D:\\sparkdata\\bigdata\\2rd_data\\ch11\\test\\part-00000"
    val arg2="D:\\sparkdata\\bigdata\\2rd_data\\ch11\\training\\part-00000"
    val arg3="./output/ch11/label"
    val arg4="23937 50 0.01 100 10"

    val numFeatures = 23937
//      arg3.toInt //特征数23937
    val partitions = 50
//      arg4.toInt //一般50-1000

    //label channal source tags
    //依次为：类别（是否点击，点击为1，没有点击为0）、频道、来源、关键词
    //样例：1 娱乐 腾讯娱乐 曲妖精|棉袄|王子文|老大爷|黑色

    // 导入训练样本和测试样本
    val training = MLUtils.loadLibSVMFile(sc,arg1,numFeatures,partitions)
    val test = MLUtils.loadLibSVMFile(sc,arg2,numFeatures,partitions)

    val lr = new LogisticRegressionWithLBFGS()

    //训练参数设置
//    23937 50 0.01 100 10
    lr.optimizer.setRegParam(0.1) //0.01
      .setNumIterations(100) //100
      .setNumCorrections(10) //10

    //训练
    val lrModel = lr.setNumClasses(2).run(training)//2分类
    lrModel.clearThreshold()

    //预测打分
    val predictionAndLabel = test.map(p=>(lrModel.predict(p.features),p.label))
    predictionAndLabel.map(x=>x._1+"\t"+x._2).repartition(1)
      .saveAsTextFile("./output/ch11tmp")
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    //计算AUC
    val str = s"the value of auc is ${metrics.areaUnderROC()}"
    println(str)
  }
}