import argparse
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import explode

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_data", type=str, help="s3 input data")
    parser.add_argument("--s3_output_data", type=str, help="s3 output data")
    parser.add_argument("--s3_metrics_data", type=str, help="s3 metrics data")
    parser.add_argument("--maxIter", type=str, help="als hyperparameter")
    parser.add_argument("--regParam", type=str, help="als hyperparameter")
    parser.add_argument("--alpha", type=str, help="als hyperparameter")
    parser.add_argument("--topN", type=str, help="number of recommendations per user")
    args = parser.parse_args()
    
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    
    parts = spark.read.format('csv').options(header='false', inferSchema='true').load(args.s3_input_data).rdd
    
    ratingsRDD = parts.map(lambda p: 
            Row(
                idcuenta=int(p[0]),
                securityid=int(p[2]),
                q=float(p[1])
            )
        )
    ratings = spark.createDataFrame(ratingsRDD)

    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build recommendation model using ALS on the training data
    als = ALS(implicitPrefs=True, maxIter=int(args.maxIter), regParam=float(args.regParam), alpha=int(args.alpha), userCol="idcuenta", itemCol="securityid",ratingCol="q", coldStartStrategy="drop")
    #Doc: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.recommendation.ALS.html
    # https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
    # https://towardsdatascience.com/build-recommendation-system-with-pyspark-using-alternating-least-squares-als-matrix-factorisation-ebe1ad2e7679

    model = als.fit(training)
    
    # Evaluate model by computing the RMSE on the test data
    predictions = model.transform(test)
    
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="q",
        predictionCol="prediction")

    RMSE = evaluator.evaluate(predictions)
    
    print("**Best Model**")# Print "Rank"
    print("  Rank:", args.topN)# Print "MaxIter"
    print("  MaxIter:", args.maxIter)# Print "RegParam"
    print("  RegParam:", args.regParam)
    print("  alpha:", args.alpha)
    print("  RMSE", RMSE)
    
    columns = ["rank","maxIter","regParam","alpha","RMSE"]
    data = [(args.topN, args.maxIter, args.regParam, args.alpha, RMSE)]
    
    rdd_metrics = spark.sparkContext.parallelize(data)
    df_metrics = rdd_metrics.toDF(columns)
    df_metrics.printSchema()
    df_metrics.show(truncate=False)

    # Generate top 10 recommendations for each user
    userRecs = model.recommendForAllUsers(int(args.topN))
    userRecs.show()
    
    # Formatting dataframe
    df = userRecs.select(userRecs.idcuenta,explode(userRecs.recommendations).alias("securityid_rating"))
    df2 = df.select(df.idcuenta, 'securityid_rating.*')
    
    # Write top 10 recommendations for each user
    df2.repartition(1).write.mode("overwrite").option("header",True).option("delimiter",",").csv(args.s3_output_data)
    df_metrics.repartition(1).write.mode("overwrite").option("header",True).option("delimiter",",").csv(args.s3_metrics_data)
    
    
if __name__ == "__main__":
    main()