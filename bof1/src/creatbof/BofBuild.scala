package creatbof

/**
  * Created by lenovo on 2015/12/4.
  */
import java.io.{InputStreamReader, BufferedReader, PrintWriter}

import com.github.fommil.netlib.{F2jBLAS, BLAS}
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import scala.collection.mutable.ArrayBuffer

//import com.massivedatascience.clusterer.{BregmanPointOps, KMeansSelector, MultiKMeansClusterer, KMeans}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
//import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
object BofBuild {
//
  def genBOF(src: String,centersPath: String ,num: Int,numClusters:Int): Unit = {
    val conf = new SparkConf().setAppName("cluster")//.setMaster("yarn-cluster")
    val sc = new SparkContext(conf)
    val data = sc.textFile(src)
    //println(data.count())
    val tagsData = data.map(f => (f.split(':')(0), f.split(':')(1))).map(s => (Vectors.dense(s._1.split(',').map(_.toDouble))))

//  val tagsData = data.map(f => (f.split(':')(0), f.split(':')(1))).filter(f => !f._1.trim.equals(""))
//    .map(s => (Vectors.dense(s._1.split(' ').map(_.toDouble)), s._2))
    val parsedData = tagsData.cache()
   // for(line <- parsedData)
      println("parsedData:"+parsedData)


    val clusterStart = System.currentTimeMillis()
   // val numClusters = 500
    val numIterations = num
    val clusters = KMeans.train(parsedData, numClusters, numIterations, 1)
    //val clusters = KMeans.train(parsedData, numClusters, numIterations, 1, "random")
    val clusterEnd = System.currentTimeMillis()
    println("cluster time : " + (clusterEnd - clusterStart))
    parsedData.unpersist()
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    //统计聚类错误的样本比例
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    //write centers to file
    val hdfsconf = new Configuration()
    val fs= FileSystem.get(hdfsconf)

    if(fs.exists(new Path(centersPath))) fs.delete(new Path(centersPath))
    val outCenters = fs.create(new Path(centersPath))
    val centersWriter = new PrintWriter(outCenters)

    clusters.clusterCenters.foreach { f =>
      centersWriter.println(f.toArray.mkString(","))
    }
    centersWriter.close()
    sc.stop()
  }


  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      System.err.println("Usage:<file>")
      System.exit(1)

    }
    BofBuild.genBOF(args(0),args(1) ,args(2).toInt,args(3).toInt)
  }

}
