# aws-mlops-fargate



This repo is used for generate aws machine learning models.

## Diagram

TODO

## Models

| Model | Description | Algorithm |
| ------ | ----------- | --------- |
| User Recommendations | Recommendations of tickers by userid | [Factorization Machines](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html) |
| Recommendations SecurityID | Recommendations of securityid by userid | [Factorization Machines](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html) |
| Top Recommendations SecurityID | Extension of Recommendations SecurityID. Will recommend top k recommendations for each user using Batch Jobs | [K-Nearest Neighbors (k-NN)](https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html) |
| ALS Recommendations SecurityID | Recommendations of securityid by userid using a non-AWS built-in algorithm [ALS](http://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) within a [PySparkProcessor](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html#pysparkprocessor) using [Sagemaker Processing Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html) | [ALS (Spark)](http://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) |

## Structure

```
.
|____.gitignore
|____ENVVARIABLES.md
|____buildpush.sh
|____CONFIG.md
|____run_task.sh
|____docs
|____requirements.txt
|____templates
| |____template.py
|____Dockerfile
|____HOWTO.md
|____src
| |____models
| | |____operations_recommendations_securityid.py
| | |____recomendation_by_wathclist.py
| | |____recommendations_securityid.py
| | |____ag_recommendations.py
| | |____als_recommendations_securityid.py
| | |____top_recommendations_securityid.py
| | |____user_recommendations.py
| | |____model.py
| |____processing_jobs
| | |____train_spark_als.py
| |____libs
| | |____utils.py
| |____app.py
|____README.md
```
