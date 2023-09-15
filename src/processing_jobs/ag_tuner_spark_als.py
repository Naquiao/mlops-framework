import argparse
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import explode
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import NumericType, StructType, StructField

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
    
    maxIter_list  = map(int, args.maxIter.split(','))
    regParam_list = map(float, args.regParam.split(','))
    alpha_list    = map(int, args.alpha.split(','))
    
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    
    parts = spark.read.format('csv').options(header='false', inferSchema='true').load(args.s3_input_data).rdd
    
    ratingsRDD = parts.map(lambda p: 
            Row(
                idpersona=int(p[0]),
                securityid=int(p[2]),
                q=float(p[1])
            )
        )
    ratings = spark.createDataFrame(ratingsRDD)

    (training, test) = ratings.randomSplit([0.8, 0.2])
    
    als = ALS(implicitPrefs=True, userCol="idpersona", itemCol="securityid",ratingCol="q", coldStartStrategy="drop")
    
    param_grid = ParamGridBuilder() \
                .addGrid(als.maxIter, maxIter_list) \
                .addGrid(als.regParam, regParam_list) \
                .addGrid(als.alpha, alpha_list) \
                .build()
                
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="q", 
            predictionCol="prediction") 
    print ("Num models to be tested: ", len(param_grid))
    
    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    
    #Fit cross validator to the 'train' dataset
    model = cv.fit(training)#Extract best model from the cv model above
    best_model = model.bestModel# View the predictions
    test_predictions = best_model.transform(test)
    RMSE = evaluator.evaluate(test_predictions)
    
    print("**Best Model**")# Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())# Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())# Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())
    print("  alpha:", best_model._java_obj.parent().getAlpha())
    print("  RMSE", RMSE)
    
    columns = ["rank","maxIter","regParam","alpha","RMSE"]
    data = [(best_model._java_obj.parent().getRank(), best_model._java_obj.parent().getMaxIter(), best_model._java_obj.parent().getRegParam(), best_model._java_obj.parent().getAlpha(), RMSE)]
    
    rdd_metrics = spark.sparkContext.parallelize(data)
    df_metrics = rdd_metrics.toDF(columns)
    df_metrics.printSchema()
    df_metrics.show(truncate=False)
    
    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(int(args.topN))
    
    # Formatting dataframe
    df = recommendations.select(recommendations.idpersona,explode(recommendations.recommendations).alias("securityid_rating"))
    df2 = df.select(df.idpersona, 'securityid_rating.*')
    
    # Write top 10 recommendations for each user
    df2.repartition(1).write.mode("overwrite").option("header",True).option("delimiter",",").csv(args.s3_output_data)
    
    df_metrics.repartition(1).write.mode("overwrite").option("header",True).option("delimiter",",").csv(args.s3_metrics_data)
    
if __name__ == "__main__":
    main()