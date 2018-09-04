import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types.{StructType, StructField, LongType, IntegerType, DoubleType}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder,CrossValidatorModel}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.feature.PCAModel
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

object Santander_Value_Predictor {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    if (args.length != 4) {
      println("I/p and O/p filepath needed")
    }
    Logger.getLogger("SVP").setLevel(Level.OFF)
    spark.sparkContext.setLogLevel("ERROR")
    val sc = spark.sparkContext
    import spark.implicits._

    // Load the train data from train.csv
    val trainData = spark.read.option("header","true").option("inferSchema", "true").csv(args(0))

    // Drop any rows with null or NAN values in it
    val filteredTrainData = trainData.na.drop(trainData.columns)

    // Drop features with only zero values
    val maxFeatureValues = filteredTrainData.groupBy().max().head
    val minFeatureValues = filteredTrainData.groupBy().min().head
    val maxDF = sc.parallelize(maxFeatureValues.toSeq.toArray.map(_.toString.toDouble)).toDF("max")
    val maxDFWithRowId = maxDF.withColumn("rowId", monotonically_increasing_id())
    val minDF = sc.parallelize(maxFeatureValues.toSeq.toArray.map(_.toString.toDouble)).toDF("min")
    val minDFWithRowId = minDF.withColumn("rowId", monotonically_increasing_id())
    val columns = filteredTrainData.columns.drop(1)
    val colDF = sc.parallelize(columns).toDF("feature")
    val colDFWithRowId = colDF.withColumn("rowId", monotonically_increasing_id())
    val table1 = colDFWithRowId.join(minDFWithRowId,"rowId").drop()
    val minMaxTable = table1.join(maxDFWithRowId,"rowId")
    val zeroValueColumns = minMaxTable.filter(row => row.getAs[Double](2) == 0 && row.getAs[Double](3) == 0).select("feature").map(_.getString(0)).collect()
    val droppedData = filteredTrainData.select(filteredTrainData.columns.filter(colName => !zeroValueColumns.contains(colName)).map(colName => new Column(colName)): _*)

    // Drop columns that are lowly correlated with target using Spearman. Use provided spearman.csv
    val colsWithCorrelation = spark.read.option("header","true").option("inferSchema", "true").csv(args(1))
    val toDropCols = colsWithCorrelation.filter(row=>row.getAs[Double](2)<0.07 && row.getAs[Double](2)> -0.07).select("col_labels").collect.map(_.getString(0))
    val subsetData = droppedData.select(droppedData.columns.filter(colName => !toDropCols.contains(colName)).map(colName => new Column(colName)): _*)


    // Assemble features into feature vector. Drop id and target from assembling
    val assembler = new VectorAssembler().setInputCols(subsetData.columns.drop(2)).setOutputCol("features")
    val assembledData = assembler.transform(subsetData)

    // Scale features using log scaler
    val featureVector = assembledData.rdd.map{row =>  row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense.toArray}
    val featureArray = featureVector.map(row=>row.map(value => scala.math.log1p(value))).collect()
    val logDF = sc.parallelize(featureArray).toDF("scaledFeatures")
    val target = spark.sqlContext.createDataFrame(logDF.rdd.zipWithIndex.map{case (row, rowid) => Row.fromSeq(row.toSeq :+ rowid)},StructType(logDF.schema.fields :+ StructField("rowid", LongType, false)))
    val source = spark.sqlContext.createDataFrame(assembledData.rdd.zipWithIndex.map{case (row, rowid) => Row.fromSeq(row.toSeq :+ rowid)},StructType(assembledData.schema.fields :+ StructField("rowid", LongType, false)))
    val scaledData = source.join(target,"rowid")
    def convertArrayToVector = udf((features: scala.collection.mutable.WrappedArray[Double]) => org.apache.spark.ml.linalg.Vectors.dense(features.toArray))
    val finalScaledData = scaledData.withColumn("scaledFeatures", convertArrayToVector($"scaledFeatures"))

    // Split train data for calculating metrics
    val Array(training, test) = finalScaledData.randomSplit(Array(0.90, 0.10))

    // Random Forest (Performs better than Linear Regression)
    val rf = new RandomForestRegressor().setMaxMemoryInMB(1024).setLabelCol("target").setFeaturesCol("scaledFeatures")

    // Parameter grid for Random Forest
    val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees,Array(50,150,250)).addGrid(rf.maxDepth,Array(10,20,30)).build()

    val evaluator = new RegressionEvaluator().setLabelCol("target").setPredictionCol("prediction").setMetricName("rmse")
    val cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5).setParallelism(4)

    val model = cv.fit(training)

    // Saving the output in a string so that it can later be saved to a text file
    var output = ""

    // RF as estimator
    val bestModel = model.bestModel.asInstanceOf[RandomForestRegressionModel]
    output += "Best model value for numTrees: "+bestModel.getNumTrees+"\n"
    output += "Best model value for maxDepth: "+bestModel.getMaxDepth+"\n\n"

    // Predict the target column on the split test data
    val result = model.transform(test)

    // LR predicts many negative values, as the competition doesn't allow negative values, converting it to positive. With RF we don't get negative predictions
    val filteredResult =  result.withColumn("prediction", when(col("prediction")<0, -col("prediction") ).otherwise(col("prediction") ))
    val finalResult = filteredResult.select("prediction","target").map{ case Row(p:Double,l:Double) => (scala.math.log1p(p),scala.math.log1p(l)) }
    val valuesAndPreds = finalResult.rdd

    // Calculating the metrics
    val metrics = new RegressionMetrics(valuesAndPreds)

    println(s"Mean Squared Logarithmic Error = ${metrics.meanSquaredError}")
    println(s"Root Mean Squared Logarithmic Error = ${metrics.rootMeanSquaredError}")
    println(s"Mean Absolute Logarithmic Error = ${metrics.meanAbsoluteError}")
    println(s"Explained Logarithmic Variance = ${metrics.explainedVariance}")

    output += "Mean Squared Logarithmic Error: "+metrics.meanSquaredError+"\n"
    output += "Root Mean Squared Logarithmic Error: "+metrics.rootMeanSquaredError+"\n"
    output += "Mean Absolute Logarithmic Error: "+metrics.meanAbsoluteError+"\n"
    output += "Explained Logarithmic variance: "+metrics.explainedVariance+"\n"

    val finalResultWithoutLog = filteredResult.select("prediction","target").map{ case Row(p:Double,l:Double) => (p,l) }
    val valuesAndPredsWithoutLog = finalResultWithoutLog.rdd
    val metricsWithoutLog = new RegressionMetrics(valuesAndPredsWithoutLog)
    println(s"R-squared = ${metricsWithoutLog.r2}")
    output += "R-Squared: "+metricsWithoutLog.r2+"\n"

    // Saving metrics to the output folder
    sc.parallelize(List(output)).repartition(1).saveAsTextFile(args(2)+"/metrics")

    // Load test data from test.csv to predict the target and submit in the competition
    val testData = spark.read.option("header","true").option("inferSchema", "true").csv(args(3))

    // Performing same pre-processing steps for test data
    val filteredTestData = testData.na.drop(testData.columns)
    val droppedTestData = filteredTestData.select(filteredTestData.columns.filter(colName => !zeroValueColumns.contains(colName)).map(colName => new Column(colName)): _*)
    val subsetTestData = droppedTestData.select(droppedTestData.columns.filter(colName => !toDropCols.contains(colName)).map(colName => new Column(colName)): _*)
    val assembledTestData = assembler.transform(droppedTestData)

    // Log scaling for test data
    val featureTestVector = assembledTestData.rdd.map{row =>  row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense.toArray}
    val featureTestArray = featureTestVector.map(row=>row.map(value => scala.math.log1p(value))).collect()
    val logTestDF = sc.parallelize(featureTestArray).toDF("scaledFeatures")
    val testTarget = spark.sqlContext.createDataFrame(logTestDF.rdd.zipWithIndex.map{case (row, rowid) => Row.fromSeq(row.toSeq :+ rowid)},StructType(logTestDF.schema.fields :+ StructField("rowid", LongType, false)))
    val testSource = spark.sqlContext.createDataFrame(assembledTestData.rdd.zipWithIndex.map{case (row, rowid) => Row.fromSeq(row.toSeq :+ rowid)},StructType(assembledTestData.schema.fields :+ StructField("rowid", LongType, false)))
    val scaledTestData = testSource.join(testTarget,"rowid")
    val finalScaledTestData = scaledTestData.withColumn("scaledFeatures", convertArrayToVector($"scaledFeatures"))

    // Predicting the test's target value using the built model
    val competitionResult = model.transform(finalScaledTestData)
    val submission = competitionResult.select($"ID".alias("ID"),$"prediction".alias("target"))

    // Prepare csv for submission
    val filteredSubmission = submission.withColumn("target", when(col("target")<0, -col("target") ).otherwise(col("target") ))
    filteredSubmission.repartition(1).write.format("csv").option("header","true").save(args(2)+"/submission")
  }
}