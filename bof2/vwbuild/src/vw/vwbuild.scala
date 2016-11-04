package vw

import java.io.{InputStreamReader, BufferedReader}
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.util.MLUtils.fastSquaredDistance
import scala.collection.mutable.ArrayBuffer
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
//import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import scala.collection.mutable.ArrayBuffer
import com.massivedatascience.clusterer.{BregmanPointOps, KMeansSelector, MultiKMeansClusterer, KMeans}
import org.apache.hadoop.conf.Configuration
//import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg._

object vwbuild {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      System.err.println("Usage:<file>")
      System.exit(1)

    }
   vwbuild.nearDuplicateRetrieval(args(0),args(1),args(2))
  }
  def nearDuplicateRetrieval(wordsPath: String, siftFeatures: String, BoFresult: String): Unit = {
    val conf = new SparkConf().setAppName("Retrieval") //.setMaster("yarn-cluster")
    val sc = new SparkContext(conf)
    val data = sc.textFile(siftFeatures)

    //read the visual words from files
    val hdfsconf = new Configuration()
    val fs = FileSystem.get(hdfsconf)

    //打开特征值文档
    if (!fs.exists(new Path(wordsPath))) //聚类中心
      throw new Exception("Centers do not exist!")
    val FileStream = fs.open(new Path(wordsPath)) //打开聚类中心，
    val br = new BufferedReader(new InputStreamReader(FileStream)) //将聚类中心转换成字符流

    var words = List[Array[Double]]() //将聚类中心存放到words中
    var line = br.readLine()
    while (line != null && !line.trim().equals("")) {
      words = line.trim().split(',').map(_.toDouble) :: words
      line = br.readLine()
    }
    fs.close()
    if (words.length != 500) {
      throw new Exception("number of initial centers error!")
    }


    //从siftFeatures中提取出特征向量
    val tagsData = data.map(f => (f.split(':')(0), f.split(':')(1))).filter(f => !f._1.trim.equals(""))
      .map(s => (Vectors.dense(s._1.split(',').map(_.toDouble)), s._2)) //将siftFeatures特征向量转换成稠密矩阵形式
    tagsData.persist()
    tagsData.count()

    val broads = System.currentTimeMillis()
    //把读到的特征向量广播出去
    val bcClusterCenters = sc.broadcast(words.toArray.map(f => Vectors.dense(f)).map(f => new VectorWithNorm(f, Vectors.norm(f, 2.0))))

    val broade = System.currentTimeMillis()
    println("broad time : " + (broade - broads))


    val bofStart = System.currentTimeMillis()
    //向量对应元素相加？什么作用
    def merge(x: Vector, y: Vector): Vector = {
      val arrX = x.toArray
      val arrY = y.toArray
      for (i <- 0 until arrX.length)
        arrX(i) = arrX(i) + arrY(i)
      Vectors.dense(arrX)
    }
    //compute siftFeatures bof /f._2：video name
    val BOF = tagsData.map(f => (f._2, new VectorWithNorm(f._1, Vectors.norm(f._1, 2.0)))) //norm计算二范数
      .map { f =>
      val thisClusterCenters = bcClusterCenters.value //brocast visual word value
    val bof = Array.fill(thisClusterCenters.length)(0.0) //初始化bof数组的长度为广播视觉词汇的个数
    val indexs = vwbuild.findClosest(thisClusterCenters, f._2, 1) //找出与之最近的视觉词汇
    var i = 0
      while (i < indexs.length) {
        bof(indexs(i).toInt) += 1 / math.pow(2, i) * (vwbuild.computeCos(f._2, thisClusterCenters(indexs(i).toInt)))
        i += 1
      }
      (f._1, Vectors.dense(bof))
    }.reduceByKey(merge)
    BOF.map(f => f._1 + ":" + f._2.toArray.mkString(",")).repartition(1).saveAsTextFile(BoFresult)
    val bofEnd = System.currentTimeMillis()
    println("train bof time : " + (bofEnd - bofStart))
    //生成bof词典结束
    bcClusterCenters.destroy()
    /*tagsData.unpersist()
    val zeroStart = System.currentTimeMillis()
    val BOFwithZero = BOF.map{f =>
      val x = f._2.toArray
      val list = new ArrayBuffer[Int]
      for(i <- 0 until x.length)
        if(x(i) > 0)
          list += i
      (f._1.split('@')(0),f._1.split('-')(1).toDouble, f._2, list.toArray)
    }
    val zeroEnd = System.currentTimeMillis()
    println(" bof zero time : " + (zeroStart - zeroEnd))

    BOF.unpersist()
    BOFwithZero.persist()
    val query = BOFwithZero.takeSample(true, 1000, 1000L)
    val retrievalStart = System.currentTimeMillis()
    def max(x: Tuple2[Int, Double], y: Tuple2[Int, Double]): Tuple2[Int, Double] = {
      if(x._2 > y._2)
        x
      else y
    }
    val bcQuery = sc.broadcast(query)
    BOFwithZero.flatMap { f =>
      val thisQuery = bcQuery.value
      val houghtT = new ArrayBuffer[(String, String, Int, Double)]
      for(q <- thisQuery) {
        val sim = computeCos(new VectorWithNorm(f._3, Vectors.norm(f._3, 2.0)), new VectorWithNorm(q._3, Vectors.norm(q._3, 2.0)))
        if (sim > 0.6) {
          val x = (q._1, f._1, ((q._2 - f._2) / 60).toInt, sim)
          houghtT += x
        }

      }
      houghtT.toSeq
    }.map(f => ((f._1, f._2, f._3), f._4)).reduceByKey(_+_).map(f => ((f._1._1, f._1._2), (f._1._3, f._2)))
      .reduceByKey(max)
      .saveAsTextFile(queryRes)
    val retrievalEnd = System.currentTimeMillis()
    print("retrieval time : " + (retrievalEnd - retrievalStart))
    sc.stop()
  }
  */
  }
    //向量标准化？?
    class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

      def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

      def this(array: Array[Double]) = this(Vectors.dense(array))

      /** Converts the vector to a dense vector. */
      def toDense = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
    }
    def computeCos(x: VectorWithNorm, y: VectorWithNorm): Double = {
      var i = 0
      var sum = 0.0
      val xArray = x.vector.toArray
      val yArray = y.vector.toArray
      while (i < x.vector.size) {
        sum += xArray(i) * yArray(i)
        i += 1
      }
      //println(sum)
      sum / (x.norm * y.norm)
    }
    //找出与广播出的视觉词汇相近的视觉单词，从而建立索引
    def findClosest(centers: TraversableOnce[VectorWithNorm], point: VectorWithNorm, k: Int): Array[Long] = {
      var bestDistance = Array.fill(k)(Double.PositiveInfinity) //最近距离初始化
      var bestIndex = Array.fill(k)(0L) //索引初始化
      var i = 0
      centers.foreach { center => //依次读取广播词汇
        val distance: Double = fastSquaredDistance(center, point)
        var j = k - 1
        while (j > -1 && distance < bestDistance(j))
          j -= 1
        if (j < k - 1) {
          var l = k - 1
          while (l > j + 1) {
            bestDistance(l) = bestDistance(l - 1)
            bestIndex(l) = bestIndex(l - 1)
            l -= 1
          }
          bestDistance(j + 1) = distance
          bestIndex(j + 1) = i
        }
        i += 1
      }
      bestIndex
    }
    //计算欧式距离
    def fastSquaredDistance(
                             v1: VectorWithNorm,
                             v2: VectorWithNorm): Double = {
      vwbuild.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
    }
    def fastSquaredDistance(
                             v1: Vector,
                             norm1: Double,
                             v2: Vector,
                             norm2: Double,
                             precision: Double = 1e-6): Double = {
      val n = v1.size
      require(v2.size == n)
      require(norm1 >= 0.0 && norm2 >= 0.0)
      val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
      val normDiff = norm1 - norm2
      var sqDist = 0.0
      /*
     * The relative error is
     * <pre>
     * EPSILON * ( \|a\|_2^2 + \|b\\_2^2 + 2 |a^T b|) / ( \|a - b\|_2^2 ),
     * </pre>
     * which is bounded by
     * <pre>
     * 2.0 * EPSILON * ( \|a\|_2^2 + \|b\|_2^2 ) / ( (\|a\|_2 - \|b\|_2)^2 ).
     * </pre>
     * The bound doesn't need the inner product, so we can use it as a sufficient condition to
     * check quickly whether the inner product approach is accurate.
     */
      val precisionBound1 = 2.0 * EPSILON * sumSquaredNorm / (normDiff * normDiff + EPSILON)
      if (precisionBound1 < precision) {
        sqDist = sumSquaredNorm - 2.0 * dot(v1, v2)
      } else if (v1.isInstanceOf[SparseVector] || v2.isInstanceOf[SparseVector]) {
        val dotValue = dot(v1, v2)
        sqDist = math.max(sumSquaredNorm - 2.0 * dotValue, 0.0)
        val precisionBound2 = EPSILON * (sumSquaredNorm + 2.0 * math.abs(dotValue)) /
          (sqDist + EPSILON)
        if (precisionBound2 > precision) {
          sqDist = Vectors.sqdist(v1, v2)
        }
      } else {
        sqDist = Vectors.sqdist(v1, v2)
      }
      sqDist
    }
    lazy val EPSILON = {
      var eps = 1.0
      while ((1.0 + (eps / 2.0)) != 1.0) {
        eps /= 2.0
      }
      eps
    }
    @transient private var _f2jBLAS: NetlibBLAS = _
    @transient private var _nativeBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }
  /**
    * dot(x, y)
    */
  def dot(x: Vector, y: Vector): Double = {
    require(x.size == y.size,
      "BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:" +
        " x.size = " + x.size + ", y.size = " + y.size)
    (x, y) match {
      case (dx: DenseVector, dy: DenseVector) =>
        dot(dx, dy)
      case (sx: SparseVector, dy: DenseVector) =>
        dot(sx, dy)
      case (dx: DenseVector, sy: SparseVector) =>
        dot(sy, dx)
      case (sx: SparseVector, sy: SparseVector) =>
        dot(sx, sy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }


  private def dot(x: DenseVector, y: DenseVector): Double = {
    val n = x.size
    f2jBLAS.ddot(n, x.values, 1, y.values, 1)
  }


  private def dot(x: SparseVector, y: DenseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.size

    var sum = 0.0
    var k = 0
    while (k < nnz) {
      sum += xValues(k) * yValues(xIndices(k))
      k += 1
    }
    sum
  }


  private def dot(x: SparseVector, y: SparseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val yIndices = y.indices
    val nnzx = xIndices.size
    val nnzy = yIndices.size

    var kx = 0
    var ky = 0
    var sum = 0.0
    // y catching x
    while (kx < nnzx && ky < nnzy) {
      val ix = xIndices(kx)
      while (ky < nnzy && yIndices(ky) < ix) {
        ky += 1
      }
      if (ky < nnzy && yIndices(ky) == ix) {
        sum += xValues(kx) * yValues(ky)
        ky += 1
      }
      kx += 1
    }
    sum
  }
}
