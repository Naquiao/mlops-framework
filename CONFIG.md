# Model Config File

## Example

```
{
    "feature_engineering": {
        "s3_query_key": "configs/recommendations-securityid.sql",
        "timeframe_days": 365,
        "dataset_format": "protobuf-sparse",
        "output_s3_encoding": "utf-8"
    },
    "train": {
        "input_prefix" : "data/recommendations-securityid",
        "output_prefix" : "models/recommendations-securityid",
        "algorithm": "factorization-machines",
        "algorithm_version": "latest",
        "hyperparameters" : {
            "predictor_type": "regressor",
            "num_factors": "64",
            "mini_batch_size": "235",
            "epochs": "140",
            "bias_init_method": "normal"
        },
        "training_datasets_content_type": "application/x-recordio-protobuf",
        "data_channel_train_key": "train",
        "data_channel_test_key": "test",
        "instance_count" : 1,
        "instance_type" : "ml.c5.xlarge"
    },
    "tuner": {
        "input_prefix" : "data/recommendations-securityid",
        "output_prefix" : "models-tuned/recommendations-securityid",
        "algorithm": "factorization-machines",
        "algorithm_version": "latest",
        "hyperparameters" : {
            "predictor_type": "regressor",
            "bias_init_method": "normal",
            "num_factors": "64"
        },
        "tuned_hyperparameters" : {
            "mini_batch_size": [225, 300],
            "epochs": [100, 140]
        },
        "objective_metric_name": "test:rmse",
        "optimization_direction": "Minimize",
        "max_jobs" : 6,
        "max_parallel_jobs" : 3,
        "training_datasets_content_type": "application/x-recordio-protobuf",
        "data_channel_train_key": "train",
        "data_channel_test_key": "test",
        "instance_count" : 1,
        "instance_type" : "ml.c5.xlarge"
    },
    "registry": {
        "model_package_group_name" : "dev-recommendations-securityid-model-package-group",
        "model_approval_status" : "Approved",
        "supported_content_types" : ["text/csv"],
        "supported_response_mime_types" : ["text/csv"]
    },
    "validate": {
        "metric" : "test:rmse",
        "optimization_direction" : "Minimize",
        "base_value": 5
    },
    "deploy": {
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
        "strategy": "MultiRecord"
    },
    "monitoring":{
        "output_prefix" : "monitoring/recommendations-securityid",
        "create_cloudwatch_dashboard": true,
        "data_quality": {
            "enabled": false,
        },
        "model_quality": {
            "enabled": false,
        }
    },
    "post_processing": {
        "knn_container_name": "dev-top-recommendations-securityid",
        "knn_model_name": "top-recommendations-securityid",
        "sfn_fargate_arn": "arn:aws:states:us-east-1:204991841662:stateMachine:dev-aws-mlops-sfn-fargate"
    },
    "enable_tuner": false,
    "enable_monitoring": false,
    "enable_registry": false,
    "enable_validate": true,
    "enable_deploy": false,
    "enable_post_processing": true,
    "dynamodb_table": "dev-models-recommendations-securityid",
    "container_name": "dev-recommendations-securityid",
    "subnet_ids": ["subnet-035da4a4ca17b72b4", "subnet-046eb519c1ab2822a", "subnet-05bae5faa2d5bf72f"],
    "security_group_ids": ["sg-00499125cafc57903"],
    "tags": [
        {"Key": "Name", "Value": "dev-default"},
        {"Key": "costCenter", "Value": "aws"},
        {"Key": "environment", "Value": "dev"},
        {"Key": "expirationDate", "Value": "01/01/2032"},
        {"Key": "owner", "Value": "aws"},
        {"Key": "prefix", "Value": "dev"},
        {"Key": "project", "Value": "aws-mlops"},
        {"Key": "role", "Value": "default"},
        {"Key": "tagVersion", "Value": "1"},
        {"Key": "created_by", "Value": "fargate"}
    ]
}
```

## Parameters

* **subnet_ids** *(list[string])*

    List of subnet ids.

* **security_group_ids** *(list[string])*

    List of security group ids. 

* **tags** *([object])*

    List of tags to be passed to the job.

    Example format:
    ```
    "tags": [
            {"Key": "Name", "Value": "dev-default"},
            {"Key": "environment", "Value": "dev"},
            {"Key": "prefix", "Value": "dev"},
            {"Key": "project", "Value": "mlops-framework"},
            {"Key": "role", "Value": "default"},
            {"Key": "tagVersion", "Value": "1"},
            ]
    ```

* **enable_tuner** *(bool)* -- **[REQUIRED]**

    Enables Hyperparameter Tuning job instead of normal Training Job.

* **enable_monitoring** *(bool)* -- **[REQUIRED]**

    Enables model monitoring.

* **enable_registry** *(bool)* -- **[REQUIRED]**

    Enables model registry.

* **enable_validate** *(bool)* -- **[REQUIRED]**

    Enables model comparison with a productive one if it exists.

* **enable_deploy** *(bool)* -- **[REQUIRED]**

    Enables model deployment.

* **enable_post_processing** *(bool)* -- **[REQUIRED]**

    Enables Post Processing.

* **container_name** *(string)*

    Fargate Model Container name

* **dynamodb_table** *(string)* -- **[REQUIRED]**

    Model Dynamodb table name

* **feature_engineering** *(object)* -- **[REQUIRED]**

    Specifies the configuration of the feature engineering step. The content of this object depends on the variables you need on your custom transformations. For example, for splitting the datasets between train and test:
    ```
    "feature_engineering": {
        "train_start_date": "2019-01-01 00:00:00",
        "train_end_date": "2020-11-01 00:00:00",
        "test_start_date": "2020-11-01 00:00:00",
        "test_end_date": "2021-02-01 00:00:00",
        "output_s3_encoding": "utf-8",
        "dataset_format": "jsonl",
    },
    ```
    ```
    def cut_df(self, df):
        train_df = df.loc[(df.index >= self.config['feature_engineering']['train_start_date']) & (
            df.index <= self.config['feature_engineering']['train_end_date'])]
        test_df = df.loc[(df.index >= self.config['feature_engineering']['test_start_date']) & (
            df.index <= self.config['feature_engineering']['test_end_date'])]
        return train_df, test_df
    ```

    + **dataset_format** *(string)* -- **[REQUIRED]**

        Dataset's file format. Accepted values: "json", "jsonl", "csv", "protobuf-sparse", "protobuf-dense"

    + **output_s3_encoding** *(string)*
        
        Dataset file's encoding. (default: 'utf-8').

* **train** *(object)*

    Specifies the configuration of the training step. Required if ‘enable_validate’: false.

    + **input_prefix** *(string)* -- **[REQUIRED]**

        S3 prefix where input data is stored, for example: “data/$model_name”

    + **output_prefix** *(string)* -- **[REQUIRED]**

        S3 prefix where training output data (such as the model.tar.gz) will be stored, for example: “models/$model_name”

    + **algorithm** *(string)* -- **[REQUIRED]**

        The name of the algorithm

    + **algorithm_version** *(string)* -- **[REQUIRED]**

        The algorithm version

    + **hyperparameters** *(object)* -- **[REQUIRED]**
        Hyperparameters to use when training the model. This must be configured based on what you define in get_train_hyperparameters.
        Example:
        ```
        "hyperparameters" : {
                    "freq": "H",
                    "prediction_length": "48",
                    "context_length": "72",
                    "epochs": "150",
                    "num_layers": "1",
                    "num_cells": "200",
                    "learning_rate": "0.001"
                },
        ```
        ```
        def get_train_hyperparameters(self):
            hyperparameters = {
                "time_freq": self.config['train']['hyperparameters']['freq'],
                "context_length": self.config['train']['hyperparameters']['context_length'],
                "prediction_length": self.config['train']['hyperparameters']['prediction_length'],
                "num_cells": self.config['train']['hyperparameters']['num_cells'],
                "num_layers": self.config['train']['hyperparameters']['num_layers'],
                "likelihood": "gaussian",
                "epochs": self.config['train']['hyperparameters']['epochs'],
                "mini_batch_size": "32",
                "learning_rate": self.config['train']['hyperparameters']['learning_rate'],
                "dropout_rate": "0.05",
                "early_stopping_patience": "10"}
        ```

    + **training_datasets_content_type** *(string)*

        MIME type of the input data (default: None).

    + **data_channel_train_key** *(string)* -- **[REQUIRED]**
        
        Train key to use in the InputDataConfig parameter in a CreateTrainingJob request. Depends on the algorithm you're using. Example: "train".

        REF: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

    + **data_channel_test_key** *(string)* -- **[REQUIRED]**

        Test/evaluation/validation key to use in the InputDataConfig parameter in a CreateTrainingJob request. Depends on the algorithm you're using. Example: "test", "evaluation", "validation".

        REF: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig


    + **Instance_count** *(int)*

    Number of Amazon EC2 instances to use for training. (default: 1)

    + **Instance_type** *(string)*

    Type of EC2 instance to use for training. (default: ‘ml.c4.xlarge’)

* **tuner** *(object)*

    Specifies the configuration of the tuner step. Required if ‘enable_tuner’: true.

    + **input_prefix** *(string)* -- **[REQUIRED]**

        S3 prefix where input data is stored, for example: “data/$model_name”

    + **output_prefix** *(string)* -- **[REQUIRED]**

        S3 prefix where hyperparameter tuner jobs output data (such as the model.tar.gz) will be stored, for example: “models-tuned/$model_name”

    + **algorithm** *(string)* -- **[REQUIRED]**

        The name of the algorithm

    + **algorithm_version** *(string)* -- **[REQUIRED]**

        The algorithm version

    + **hyperparameters** *(object)* -- **[REQUIRED]**

        Static Hyperparameters to use when executing the hyperparameter tuning job. This must be configured based on what you define in get_tuner_hyperparameters.

    + **tuned_hyperparameters** *(object)* -- **[REQUIRED]**

        Dynamic range Hyperparameters to use when executing the hyperparameter tuning job. This must be configured based on what you define in get_tuner_hyperparameters. Example:
        ```
        "hyperparameters" : {
                    "freq": "H",
                    "prediction_length": "48",
                    "context_length": "72"
                },
                "tuned_hyperparameters" : {
                    "num_cells": [30, 200],
                    "num_layers": [1, 2],
                    "epochs": [100, 200],
                    "learning_rate": [1e-5, 0.1],
                    "dropout_rate": [0.1, 0.9]
                },
        ```
        ```
        def get_tuner_hyperparameters(self):
            static_hyperparameters = {
                "time_freq": self.config['tuner']['hyperparameters']['freq'],
                "context_length": self.config['tuner']['hyperparameters']['context_length'],
                "prediction_length": self.config['tuner']['hyperparameters']['prediction_length'],
                "likelihood": "gaussian",
                "mini_batch_size": "32",
                "early_stopping_patience": "10"}

            tuned_hyperparameters = {
                "num_cells": IntegerParameter(
                    self.config['tuner']['tuned_hyperparameters']['num_cells'][0],
                    self.config['tuner']['tuned_hyperparameters']['num_cells'][1]),
                "num_layers": IntegerParameter(
                    self.config['tuner']['tuned_hyperparameters']['num_layers'][0],
                    self.config['tuner']['tuned_hyperparameters']['num_layers'][1]),
                "epochs": IntegerParameter(
                    self.config['tuner']['tuned_hyperparameters']['epochs'][0],
                    self.config['tuner']['tuned_hyperparameters']['epochs'][1]),
                "learning_rate": ContinuousParameter(
                    self.config['tuner']['tuned_hyperparameters']['learning_rate'][0],
                    self.config['tuner']['tuned_hyperparameters']['learning_rate'][1]),
                "dropout_rate": ContinuousParameter(
                    self.config['tuner']['tuned_hyperparameters']['dropout_rate'][0],
                    self.config['tuner']['tuned_hyperparameters']['dropout_rate'][1])}

            return static_hyperparameters, tuned_hyperparameters
        ```

    + **objective_metric_name** *(int)* -- **[REQUIRED]**

        Name of the metric for evaluating training jobs.

    + **optimization_direction** *(string)* -- **[REQUIRED]**

        The type of the objective metric for evaluating training jobs. This value can be either ‘Minimize’ or ‘Maximize’ (default: ‘Maximize’).

    + **max_jobs** *(int)*

        Maximum total number of training jobs to start for the hyperparameter tuning job (default: 1).

    + **max_parallel_jobs** *(dict)*

        Maximum number of parallel training jobs to start (default: 1).
    
    + **training_datasets_content_type** *(string)*

        MIME type of the input data (default: None).

    + **data_channel_train_key** *(string)* -- **[REQUIRED]**
        
        Train key to use in the InputDataConfig parameter in a CreateTrainingJob request. Depends on the algorithm you're using. Example: "train".

        REF: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

    + **data_channel_test_key** *(string)* -- **[REQUIRED]**

        Test/evaluation/validation key to use in the InputDataConfig parameter in a CreateTrainingJob request. Depends on the algorithm you're using. Example: "test", "evaluation", "validation".

        REF: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

    + **instance_count** *(int)*

        Number of Amazon EC2 instances to use for training. (default: 1)

    + **instance_type** *(string)*

        Type of EC2 instance to use for training. (default: ‘ml.c4.xlarge’)

    + **instance_type** *(dict)*

        The configuration of Amazon Elastic Block Store (Amazon EBS) attached to each instance as defined by InstanceType .

* **registry** *(object)*

    Specifies the configuration of the model registry step. Required if ‘enable_registry’: true.

    + **model_package_group_name** *(string)* -- **[REQUIRED]**

        The name of the model group that this model version belongs to.

    + **model_approval_status** *(string)* -- **[REQUIRED]**

        Whether the model is approved for deployment. Accepted values: "Approved", "Rejected", "PendingManualApproval"

    + **supported_content_types** *(list[string])* -- **[REQUIRED]**

        The supported MIME types for the input data. Example: ["text/csv"]

    + **supported_response_mime_types** *(list[string])* -- **[REQUIRED]**

        The supported MIME types for the output data. Example: ["text/csv"]

* **validate** *(object)*

    Specifies the configuration of the model validation step. Required if ‘enable_validate’: true.

    + **objective_metric_name** *(int)* -- **[REQUIRED]**

        Name of the metric for validating models.

    + **optimization_direction** *(string)* -- **[REQUIRED]**

        The type of the objective metric for comparing models. This value can be either ‘Minimize’ or ‘Maximize’

* **deploy** *(object)* -- **[REQUIRED]**

    Specifies the configuration of the model deployment step.

    + **initial_instance_count** *(int)*

        Minimum number of EC2 instances to deploy to an endpoint for prediction. (default: 1)

    + **instance_type** *(string)*

        Type of EC2 instance to deploy to an endpoint for prediction, for example, ‘ml.c4.xlarge’. (default: "ml.m5.large")

    + **data_capture_config** *(object)*

        Specifies configuration related to Endpoint data capture for use with Amazon SageMaker Model Monitoring. Default: None.

        - **output_prefix (string)** -- **[REQUIRED]**
            
            S3 prefix where data capture data is stored, for example: “inference_data_capture/$model_name”

        - **sampling_percentage** *(int)*

            The percentage of data to sample. Must be between 0 and 100. (default: 20)

        - **capture_options** *(list)*

            Must be a list containing any combination of the following values: “REQUEST”, “RESPONSE”. Denotes which data to capture between request and response. (default: [“REQUEST”, “RESPONSE”])

    + **enable_autoscaling** *(bool)*

        Enables Application Autoscaling policy on Sagemaker Endpoint. (default: false)

    + **autoscaling** *(object)*

        Application Autoscaling configuration (Target Tracking). Required if *enable_autoscaling* is true.

        - **min_capacity** *(int)* -- **[REQUIRED]**

            The minimum value that you plan to scale in to. When a scaling policy is in effect, Application Auto Scaling can scale in (contract) as needed to the minimum capacity limit in response to changing demand. 

        - **max_capacity** *(int)* -- **[REQUIRED]**

            The maximum value that you plan to scale out to. When a scaling policy is in effect, Application Auto Scaling can scale out (expand) as needed to the maximum capacity limit in response to changing demand.

        Application Autoscaling policy specifics must be set in each model class under the **get_autoscaling_policies()** method. Example
        ```
        def get_autoscaling_policies(self):
            policies = [{
                        'TargetValue': 80.0,
                        'CustomizedMetricSpecification': {
                            'MetricName': 'CPUUtilization',
                            'Namespace': '/aws/sagemaker/Endpoints',
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': self.config["model_id"]
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': 'AllTraffic'
                                }
                            ],
                            'Statistic': 'Average',
                            'Unit': 'Percent'
                        },
                        'ScaleOutCooldown': 300,
                        'ScaleInCooldown': 300,
                        'DisableScaleIn': False
                    },
                    {
                        'TargetValue': 80.0,
                        'CustomizedMetricSpecification': {
                            'MetricName': 'MemoryUtilization',
                            'Namespace': '/aws/sagemaker/Endpoints',
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': self.config["model_id"]
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': 'AllTraffic'
                                }
                            ],
                            'Statistic': 'Average',
                            'Unit': 'Percent'
                        },
                        'ScaleOutCooldown': 300,
                        'ScaleInCooldown': 300,
                        'DisableScaleIn': False
                    }]
            return policies
        ```

* **monitoring** *(object)*

    Specifies the configuration of the model monitoring step. Required if ‘enable_monitoring’: true.

    + **output_prefix** *(string)* -- **[REQUIRED]**

        S3 prefix where the monitoring output data is stored, for example: “monitoring/$model_name”

    + **create_cloudwatch_dashboard** *(bool)*

        Whether to generate a CloudWatch dashboard or not. (default: true)

    + **data_quality** *(object)*

        Configurations needed for data quality monitoring. Currently only supported by XGBoost.

    + **model_quality** *(object)*

        Configurations needed for model quality monitoring. Currently only supported by XGBoost.

* **post_processing** *(object)* -- **[REQUIRED]**

    Specifies the configuration of the post_processing step. The content of this object depends on what you need to do. For example, if we want to use a second algorithm like factorization machines and k-NN for our model, we can call a new step function to trigger the second pipeline from factorization-machines pipeline -> k-NN pipeline:
    ```
    "post_processing": {
        "knn_container_name": "${knn_container_name}",
        "knn_model_name": "${knn_model_name}",
        "sfn_fargate_arn": "${sfn_fargate_arn}"
    }
    ```
