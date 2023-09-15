# balanz-mlops-fargate

https://github.com/Naquiao/mlops-framework/blob/main/docs/flow-diagram.png
![Overview]([http://url/to/img.png](https://github.com/Naquiao/mlops-framework/blob/main/docs/flow-diagram.png))
This repo is used for generate Balanz machine learning models.

For more information about how the infra behind this works, please refer to: [balanz-mlops-terraform/README](https://www.balanz.com/bitbucket/projects/BAL-DL/repos/balanz-mlops-terraform/browse/README.md)

For more information about how to run and troubleshoot models, please refer to: [balanz-mlops-terraform/USAGE](https://www.balanz.com/bitbucket/projects/BAL-DL/repos/balanz-mlops-terraform/browse/USAGE.md)

For more information about the Model Config file, please refer to: [CONFIG.md](https://www.balanz.com/bitbucket/projects/BAL-DL/repos/balanz-mlops-fargate/browse/CONFIG.md)

For more information about the Environment Variables, please refer to: [ENVVARIABLES.md](https://www.balanz.com/bitbucket/projects/BAL-DL/repos/balanz-mlops-fargate/browse/ENVVARIABLES.md)

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
