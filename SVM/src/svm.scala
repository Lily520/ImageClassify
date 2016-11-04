import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Administrator on 2016/4/23.
  */

object svm {
  def main(args:Array[String]) : Unit = {
    //load data
    System.setProperty("hadoop.home.dir", "C:\\spark\\hadoop-2.6.0\\hadoop-2.6.0")
    val conf = new SparkConf().setAppName("SVM")
    //    println("11111")
    val sc = new SparkContext(conf)
    val data = MLUtils.loadLibSVMFile(sc, "C:\\VOC2010\\1.txt")
//    val data = MLUtils.loadLibSVMFile(sc, "hdfs://tseg0:9010/user/tseg/zll/2007_000027.txt")

    println("3333")
    //split data into training and test
    val splits = data.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splits(0).cache()
    val test = splits(1).cache()

    //run training algorithm to bulid the model
    val numIterations = 100
    val model = SVMWithSGD.train(training,numIterations)


    //clear the default threshold
    model.clearThreshold()
     println("I am hungty!!!")
    //compute raw scores on the test set
    val scoreAndLabels = test.map{point =>
      val score = model.predict(point.features)
      (score,point.label)
    }

    //get evaluation metrics
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)

    val trainErr = scoreAndLabels.filter(r => r._1 != r._2).count().toDouble / data.count
    println("准确率Accuracy：" + (1.0 - trainErr) * 100 + "%")

    sc.stop()
  }

}
