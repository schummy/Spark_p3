
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.optimization.L1Updater
/**
  * Created by Andrii_Krasnolob on 3/31/2016.
  */
object SparkML {
  var inputDataDir: String = "dataInput/"
  val dataFileName:String = "all_data.csv"
  val inputDataFileName:String = "input_data.csv"
  val modelSaveDirectory: String = "tmpModelPath"


  def main(args:Array[String]): Unit = {

    val conf: SparkConf = initConfig()

    val sc = new SparkContext(conf)
    LogManager.getRootLogger.setLevel(Level.ERROR)
    val sameModel:RandomForestModel =
    {
      try {
        RandomForestModel.load(sc, modelSaveDirectory)
      }
      catch {
        case e: Exception =>
          LogManager.getLogger("SparkML").warn("Can't load existing model. Will try to build new one.")
          val data = sc.textFile(inputDataDir + dataFileName)
          val parsedData: RDD[LabeledPoint] = prepareData(data)
          val splits = parsedData.randomSplit(Array(0.7, 0.3)) //, 13l)
          val (trainingData, testData) = (splits(0), splits(1))

          var model = buildModel(trainingData)
          model.save(sc, modelSaveDirectory)
          model

      }
    }
    val inputData = sc.textFile(inputDataDir+inputDataFileName)
    val parsedInputData: RDD[LabeledPoint] = prepareData(inputData)
    val scoreAndLabelsForInputData: RDD[(Double, Double)] = testModel(parsedInputData, sameModel)
    calculateMetrics(scoreAndLabelsForInputData)
   // predictProbability(parsedInputData,sameModel).foreach(println)
    predict(parsedInputData,sameModel).foreach(a => {println (f"Probability[${a._1}%3d] = ${a._2}")})
  }

  def initConfig(): SparkConf = {
    if (System.getProperty("os.name") == "Windows 7") {
      System.setProperty("hadoop.home.dir", "C:\\BigData\\Hadoop")
      //inputDataDir = "D:\\Share\\Spark_Basics_p3\\Spark_Basics_p3_data\\session.dataset\\"

    } else {
      //inputDataDir = "/Users/user/bigData/EPAM_hadoop_training/Spark_Basics_p3_data/session.dataset/"
    }

    new SparkConf()
      .setAppName("SparkML")
      .setMaster("local")
      .set("spark.executor.memory", "2048m")
      .set("spark.driver.memory", "2048m")
  }

  def calculateMetrics(scoreAndLabels: RDD[(Double, Double)]): Unit = {
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)
  }

  def testModel(testData: RDD[LabeledPoint], model: RandomForestModel): RDD[(Double, Double)] = {

    testData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

  }

  def predictProbability(points: RDD[LabeledPoint], model: RandomForestModel) = {
    val numTrees = model.trees.length
    val trees = points.sparkContext.broadcast(model.trees)
    points.map { point =>
      trees.value
        .map(_.predict(point.features))
        .sum / numTrees
    }
  }


  def predict(points: RDD[LabeledPoint], model: RandomForestModel) = {
    val numTrees = model.trees.length
    val trees = points.sparkContext.broadcast(model.trees)
    var i = 0
    points.map { point =>
      ( {i += 1; i},
        trees.value.map(_.predict(point.features)).sum / numTrees
        //, model.predict(point.features)
        //, point.label
        )
    }
  }


  def buildModel(trainingData: RDD[LabeledPoint]): RandomForestModel = {

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 15
    val maxBins = 100
    RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

     /* val svmAlg = new SVMWithSGD()
    svmAlg.optimizer.
      setNumIterations(20). //0.533987899956109
      setRegParam(0.1).
      setUpdater(new L1Updater)

    val model = svmAlg.run(trainingData)*/
  }

  def prepareData(data: RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>
      val parts = line.split(',')
      var featureVector: linalg.Vector = buildFeatureVector(parts.tail)
      LabeledPoint(parts(0).toDouble, featureVector)
    }
  }


  def buildFeatureVector(features:Array[String]): linalg.Vector = {
    var featureSeq: Seq[(Int, Double)] = Seq[(Int, Double)]()
    var activeFeaturesList: List[Int] = List(15, 16, 17, 18, 38, 37) // Selected based on PCA analysis


    var featureLength: Int = activeFeaturesList.length

    var featurePosition = 0
    for (i <- features.indices) {
      if (activeFeaturesList.contains(i)){
        val featureInfo = preprocessData(features, i, featureSeq, featurePosition)
        featureSeq = featureInfo._1
        featurePosition = featureInfo._2
      }
    }
    Vectors.sparse(featureLength, featureSeq)
  }

  def preprocessData(features:Array[String]
                     , i:Int
                     , featureSeq:Seq[(Int, Double)]
                     , featurePosition:Int):
  (Seq[(Int, Double)], Int) = {
    var value: Double = 0.0
    if (!features(i).isEmpty) {
      if (features(i) == "NaN") {
        if (i == 8) value = 31.0
        if (i == 9) value = 12.0
        if (i == 10) value = 5.0
        if (i == 11) value = 2.0
        if (i == 12) value = 10.0
        if (i == 38) value = 0.0
        if (i == 49) value = 0.0
      }
      else {
        value = features(i).toDouble

        if (i == 0)
          value = Math.round((value - 21) / 10).toDouble
        else if (i == 14 || i == 32 || i == 34) {
          if (value < 5000) value = 0.0
          else if (value < 10000) value = 1.0
          else if (value < 20000) value = 2.0
          else if (value < 50000) value = 3.0
          else value = 4.0
        }
        else if (i == 37) {
          if (value / 12 > features(0).toDouble) value = 0.0
          else value = math.round(value / 12).toDouble
        }
        else if (i == 47 || i == 48) {
          if (value > 0) value = 1.0
          else value = 0.0
        }
      }
      ( featureSeq :+(featurePosition, value),  featurePosition +1)
    }
    else {
      (featureSeq, featurePosition)
    }
  }
}