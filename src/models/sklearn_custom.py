import libs.utils as utils
import logging
import pandas as pd
import numpy as np
import tarfile
from sagemaker import image_uris
from sagemaker import get_execution_role
import sagemaker
import os
import boto3
import awswrangler as wr
from datetime import datetime
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.tuner import IntegerParameter
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.network import NetworkConfig
from sklearn.model_selection import train_test_split
from models.model import Model
from decimal import Decimal


class sklearn_custom(Model):

    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        # FEATURE ENG
        #Aca cambiar por parametros de qquery de terraform
        self.query = utils.read_file_from_bucket(
            self.config['bucket_models'],
            self.config['feature_engineering']['s3_query_key'])
            
        self.train_dataset_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/iris_train_dataset.csv'
        self.test_dataset_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/iris_test_dataset.csv'
        self.dataset_directory = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset'
        #Bucket del output
        self.sklearn_output_bucket = os.environ.get('SKLEARN_OUTPUT_BUCKET')
        self.sklearn_output_s3_key = f'{self.config["model_name"]}/{self.config["model_id"]}/output/'
        self.sklearn_metrics_s3_key = f'{self.config["model_name"]}/{self.config["model_id"]}/metrics'
        self.sklearn_output_s3_uri = f's3://{self.sklearn_output_bucket}/{self.sklearn_output_s3_key}'
        self.sklearn_metrics_s3_uri = f's3://{self.sklearn_output_bucket}/{self.sklearn_metrics_s3_key}'

    def data_preparation(self, df):

        logging.info("#############DATA PREPARATION#############")

        df.drop(0, axis=0,inplace=True)
        df.drop(columns=['id'], axis=1,inplace=True)
        df.reset_index(inplace=True, drop=True)
        class_labels = ['Iris-versicolor','Iris-setosa','Iris-virginica']
        df.replace(class_labels, [0, 1, 2], inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(
                df.drop("sepal_width", axis=1), df["sepal_width"], test_size=0.2, random_state=0
            )
        X_train['target'] = y_train
        X_test['target'] = y_test

        logging.info(f'Xytrain and Xytest')

        encoding = self.config['feature_engineering']['output_s3_encoding'] if 'output_s3_encoding' in self.config['feature_engineering'] else 'utf-8'

        # saves dataset
        upload = utils.put_obj_in_bucket(
            X_train,
            encoding,
            self.config['bucket_models'],
            self.train_dataset_key ,
            'csv',
            True)
        logging.info(f'Putting df obj: {upload}')

        upload = utils.put_obj_in_bucket(
            X_test ,
            encoding,
            self.config['bucket_models'],
            self.test_dataset_key,
            'csv',
            True)
        logging.info(f'Putting df obj: {upload}')

        output = {
            "s3_train_dataset_file": "s3://{}/{}".format(self.config['bucket_models'], self.train_dataset_key),
            "s3_test_dataset_file": "s3://{}/{}".format(self.config['bucket_models'], self.test_dataset_key ),
            "end_date": str(datetime.now())
        }

        return output
        
    def train(self):
        step = "train"
        logging.info("#############STARTING MODEL TRAINING#############")
        sm_boto3 = boto3.client("sagemaker")

        # Parameters for creating the Estimator
        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()
        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"
        framework_version = self.config[step]["framework_version"] if 'framework_version' in self.config[step] else "0.23-1"
        framework_name = self.config[step]["framework_name"] if 'framework_name' in self.config[step] else "sklearn"
        python_version = self.config[step]["python_version"] if 'python_version' in self.config[step] else "py3"
        metric_definitions = self.config[step]["metric_definitions"] if 'metric_definitions' in self.config[step] else [{"Name": "median-AE", "Regex": "AE-at-50th-percentile: ([0-9.]+).*$"}]


        #First compress the code and send to S3 - tar.gz due to compliance
        source = "source.tar.gz"
        tar = tarfile.open(source, "w:gz")
        tar.add("script_mode/sklearn_train.py", os.path.basename("sklearn_train.py"))
        tar.close()

         # Define paths to upload the training script
        sklearn_script_mode_s3_key = f"batch-inferences/{self.config['model_name']}/{self.config['model_id']}/script_mode/sklearn_train.tar.gz"
        sklearn_script_mode_s3_uri = f"s3://{self.config['bucket_models']}/{sklearn_script_mode_s3_key}"

        #uploading
        s3 = boto3.resource('s3')
        model_bucket = s3.Bucket(self.config['bucket_models']).upload_file(source, sklearn_script_mode_s3_key)
        logging.info(f'Training script succesfuly uploaded to: {sklearn_script_mode_s3_uri}')

        #Define train and test datasets path
        trainpath = "s3://{}/{}".format(self.config['bucket_models'], self.train_dataset_key)
        testpath = "s3://{}/{}".format(self.config['bucket_models'], self.test_dataset_key)

        logging.info(f'Train path: {trainpath}')
        logging.info(f'Test path: {testpath}')

        #Retrieving the image of the Script-Mode Container
        training_image = image_uris.retrieve(
            framework=framework_name,
            region=self.config["region"],
            version=framework_version,
            py_version=python_version,
            instance_type=instance_type,
        )
        logging.info(f'Training image retrieved succesfully : {training_image}')

        #TODO: parsear a objeto de terraform
        hyperparameters={
                "n-estimators": 100,
                "min-samples-leaf": 3
                }

        #Estimator
        sklearn_estimator = SKLearn(
            entry_point='sklearn_train.py',
            source_dir =sklearn_script_mode_s3_uri,
            output_path = self.sklearn_output_s3_uri,
            image_uri = training_image,
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            base_job_name=self.config["model_id"],
            subnets=subnet_ids,
            security_group_ids=security_group_ids,
            tags=tags,
            metric_definitions=metric_definitions,
            hyperparameters=hyperparameters,
                )


        logging.info(f'Estimator succesfully created: {sklearn_estimator}')

        # Estimator training
        sklearn_estimator.fit({"train": trainpath, "test": testpath})
        
        # Adding Billable Time of the Training Job
        client = boto3.client('sagemaker')

        response = client.describe_training_job(
            TrainingJobName=sklearn_estimator.latest_training_job.name
        )

        training_billable_time = response.get('BillableTimeInSeconds')
        output = {
            "image_uri": training_image,
            "hyperparameters": hyperparameters,
            "job_name": sklearn_estimator.latest_training_job.name,
            "s3_model_data": sklearn_estimator.model_data,
            "end_date": str(datetime.now()),
            "metrics": metric_definitions,
            "billable_time": training_billable_time
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])



        logging.info(f'######### TRAINING COMPLETED ######### ')


        
    def tuner(self):
        step = "tuner"
        logging.info("#############STARTING MODEL TUNING#############")
        sagemaker_session = sagemaker.Session()
        sm_boto3 = boto3.client("sagemaker")

        # Parameters for creating the Estimator
        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()
        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"
        framework_version = self.config[step]["framework_version"] if 'framework_version' in self.config[step] else "0.23-1"
        framework_name = self.config[step]["framework_name"] if 'framework_name' in self.config[step] else "sklearn"
        python_version = self.config[step]["python_version"] if 'python_version' in self.config[step] else "py3"
        metric_definitions = self.config[step]["metric_definitions"] if 'metric_definitions' in self.config[step] else [{"Name": "median-AE", "Regex": "AE-at-50th-percentile: ([0-9.]+).*$"}]
        

        #First compress the code and send to S3 - tar.gz due to compliance
        source = "source.tar.gz"
        tar = tarfile.open(source, "w:gz")
        tar.add("script_mode/sklearn_train.py", os.path.basename("sklearn_train.py"))
        tar.close()

         # Define paths to upload the training script
        sklearn_script_mode_s3_key = f"batch-inferences/{self.config['model_name']}/{self.config['model_id']}/script_mode/sklearn_train.tar.gz"
        sklearn_script_mode_s3_uri = f"s3://{self.config['bucket_models']}/{sklearn_script_mode_s3_key}"

        #uploading
        s3 = boto3.resource('s3')
        model_bucket = s3.Bucket(self.config['bucket_models']).upload_file(source, sklearn_script_mode_s3_key)
        logging.info(f'Training script succesfuly uploaded to: {sklearn_script_mode_s3_uri}')

        #Define train and test datasets path
        trainpath = "s3://{}/{}".format(self.config['bucket_models'], self.train_dataset_key)
        testpath = "s3://{}/{}".format(self.config['bucket_models'], self.test_dataset_key)

        logging.info(f'Train path: {trainpath}')
        logging.info(f'Test path: {testpath}')

        #Retrieving the image of the Script-Mode Container
        training_image = image_uris.retrieve(
            framework=framework_name,
            region=self.config["region"],
            version=framework_version,
            py_version=python_version,
            instance_type=instance_type,
        )
        logging.info(f'Training image retrieved succesfully : {training_image}')

        #TODO: parsear a objeto de terraform
        hyperparameters={
                "n-estimators": 100,
                "min-samples-leaf": 3
                }

        #Estimator
        sklearn_estimator = SKLearn(
            entry_point='sklearn_train.py',
            source_dir =sklearn_script_mode_s3_uri,
            output_path = self.sklearn_output_s3_uri,
            image_uri = training_image,
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            base_job_name=self.config["model_id"],
            subnets=subnet_ids,
            security_group_ids=security_group_ids,
            tags=tags,
            metric_definitions=metric_definitions,
            hyperparameters=hyperparameters,
                )


        logging.info(f'Estimator succesfully created: {sklearn_estimator}')

        max_jobs = self.config[step]["max_jobs"] if 'max_jobs' in self.config[step] else 1
        max_parallel_jobs = self.config[step]["max_parallel_jobs"] if 'max_parallel_jobs' in self.config[step] else 1
        objective_type = self.config[step]["objective_type"] if 'objective_type' in self.config[step] else "Minimize"
        objective_metric_name = self.config[step]["objective_metric_name"] if 'objective_metric_name' in self.config[step] else "median-AE"
        

        # Define exploration boundaries
        hyperparameter_ranges = {
            "n-estimators": IntegerParameter(20, 100),
            "min-samples-leaf": IntegerParameter(2, 6),
        }

        # create sklearn tuner
        sklearn_tuner = sagemaker.tuner.HyperparameterTuner(
            estimator=sklearn_estimator,
            hyperparameter_ranges=hyperparameter_ranges,
            base_tuning_job_name=self.config["model_id"],
            objective_type=objective_type,         # o Maximizar segun el caso
            objective_metric_name=objective_metric_name,  # Métrica que definimos en el script!!!!
            metric_definitions=metric_definitions, 
            max_jobs= max_jobs, 
            max_parallel_jobs= max_parallel_jobs,
        )

        sklearn_tuner.fit({"train": trainpath, "test": testpath})

        #Me quedo con el mejor job del HPTunning
        sklearn_tuner.best_training_job()

        artifact_s3_uri = sm_boto3.describe_training_job(
            TrainingJobName=sklearn_tuner.best_training_job())["ModelArtifacts"]["S3ModelArtifacts"]

        logging.info(f'Best tunning job at : {artifact_s3_uri}')

        response = sm_boto3.describe_training_job(
            TrainingJobName=sklearn_tuner.best_training_job()
        )

        logging.info(f'describe training job: {response}')

        logging.info(f'metrics list: {response["FinalMetricDataList"]}')

        metric_dict = {}

        # removing timestamp from FinalMetricDataList
        for metric in response["FinalMetricDataList"]:
            metric_dict[metric["MetricName"]] = Decimal(metric["Value"])

        logging.info(f'metric_dict: {metric_dict}')

        # Adding total billing time of tuner step
        #tuning_jobs_df = sklearn_tuner.analytics().dataframe()
        #tuning_billable_time = tuning_jobs_df['TrainingElapsedTimeSeconds'].sum()

        output = {
            "image_uri": training_image,
            "job_name": sklearn_tuner.best_training_job(),
            "best_job_s3_uri": artifact_s3_uri,
            "hyperparameters": sklearn_tuner.best_estimator().hyperparameters(),
            "s3_model_data": sklearn_tuner.best_estimator().model_data,
            "end_date": str(datetime.now()),
            "metrics": metric_dict
            #"billable_time" : tuning_billable_time
        }

        self.config["dynamodb_item"]["steps"]["tuner"] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def registry(self):
        logging.info(
            "Registry not available for als_recommendations_securityid")

    def validate(self):
        logging.info(
            "Validation not implemented for als_recommendations_securityid")
        #TODO: self.config['dynamodb_item']['steps']['train']['metrics']['rmse'] can be used to measure the model
        
    def deploy(self):
        step = "deploy"
        logging.info("#############STARTING DEPLOY#############")

        training_job_name = self.config["dynamodb_item"]["steps"]["tuner"]["job_name"]
        initial_instance_count = self.config[step][
            "initial_instance_count"] if 'initial_instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.m5.large"
        model_name = self.config["model_id"][:63]
        endpoint_name = self.config["model_id"][:63]
        model_data = self.config["dynamodb_item"]["steps"]["tuner"]["best_job_s3_uri"]
        #data_capture_config = self.get_data_capture_config()
        tags = self.get_model_tags()
        framework_version = self.config[step]["framework_version"] if 'framework_version' in self.config[step] else "0.23-1"
        framework_name = self.config[step]["framework_name"] if 'framework_name' in self.config[step] else "sklearn"
        python_version = self.config[step]["python_version"] if 'python_version' in self.config[step] else "py3"

        #First compress the code and send to S3 - tar.gz due to compliance
        source = "source.tar.gz"
        tar = tarfile.open(source, "w:gz")
        tar.add("script_mode/sklearn_train.py", os.path.basename("sklearn_train.py"))
        tar.close()

         # Define paths to upload the training script
        sklearn_script_mode_s3_key = f"batch-inferences/{self.config['model_name']}/{self.config['model_id']}/script_mode/sklearn_train.tar.gz"
        sklearn_script_mode_s3_uri = f"s3://{self.config['bucket_models']}/{sklearn_script_mode_s3_key}"

        #uploading
        s3 = boto3.resource('s3')
        model_bucket = s3.Bucket(self.config['bucket_models']).upload_file(source, sklearn_script_mode_s3_key)
        logging.info(f'Training script succesfuly uploaded to: {sklearn_script_mode_s3_uri}')

        #Retrieving the image of the Script-Mode Container
        training_image = image_uris.retrieve(
            framework=framework_name,
            region=self.config["region"],
            version=framework_version,
            py_version=python_version,
            instance_type=instance_type,
        )
        logging.info(f'Training image retrieved succesfully : {training_image}')

        # Creating SKLearnModel with the best training job
        model = SKLearnModel(
            model_data=model_data,
            image_uri = training_image,
            role= self.config['role'],
            entry_point='sklearn_train.py',
            source_dir = sklearn_script_mode_s3_uri,
            code_location = self.sklearn_output_s3_uri,
            framework_version=framework_version,
        )

        # Deploying the SKLearn model created
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name = endpoint_name,
            model_name = model_name,
            tags=tags
            )

        output = {
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "instance_type": instance_type,
            "initial_instance_count": initial_instance_count,
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

        message = {
            "New Model Deployed": self.config['model_name'],
            "Model ID": self.config['model_id'],
            "Endpoint ID": endpoint_name
        }

        utils.send_sns_notification(
            self.config['sns_topic_arn'], message)
    
    def monitoring(self):
        logging.info(
            "Monitoring not available for als_recommendations_securityid")
        
    def delete(self):
        logging.info(
            "Delete not available for als_recommendations_securityid")

    def post_processing(self):
        step = "post_processing"
        logging.info(
            "#############STARTING POST PROCESSING#############")
        
        dataset_recommendations = wr.s3.read_csv(self.config["dynamodb_item"]["steps"]["train"]["s3_als_output"])
        dataset_original = wr.s3.read_csv(self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_dataset_file"])
        
        dataset_recommendations.rename(columns={"securityid": "security_id_num"},inplace=True)
        dataset_original = dataset_original[['security_id_num','securityid','grupo']].drop_duplicates()
        df = dataset_recommendations.merge(dataset_original,on='security_id_num',how='left')
        df = df[['idcuenta','securityid','grupo','rating']]
        
        encoding = self.config['post_processing']['output_s3_encoding'] if 'output_s3_encoding' in self.config['post_processing'] else 'utf-8'
        results_key = f"{self.config['post_processing']['output_prefix']}/{self.config['model_id']}/results/results.csv"
        s3_results_uri = f"s3://{self.config['bucket_models']}/{results_key}"
        common_results_key = f"{self.config['post_processing']['output_prefix']}/common/results/results.csv"
        s3_common_results_uri = f"s3://{self.config['bucket_models']}/{common_results_key}"
                
        # saves dataset in model_id directory
        upload = utils.put_obj_in_bucket(
            df,
            encoding,
            self.config['bucket_models'],
            results_key,
            'csv',
            True)
        logging.info(f'Putting df obj: {upload}')
        
        # saves dataset in a common directory
        upload = utils.put_obj_in_bucket(
            df,
            encoding,
            self.config['bucket_models'],
            common_results_key,
            'csv',
            True)
        logging.info(f'Putting df obj: {upload}')
        
        output = {
            "end_date": str(datetime.now()),
            "s3_results_uri": s3_results_uri,
            "s3_common_results_uri": s3_common_results_uri
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

        message = {
                "Model Name": self.config["model_name"],
                "Model ID": self.config['model_id'],
                "S3 Results URI": s3_results_uri,
                "S3 Common Results URI": s3_common_results_uri
            }
        utils.send_sns_notification(
            self.config['sns_topic_arn'], message)