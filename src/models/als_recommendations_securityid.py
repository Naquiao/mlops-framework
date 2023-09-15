import libs.utils as utils
import logging
import pandas as pd
import numpy as np
import sagemaker
import os
import boto3
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.network import NetworkConfig
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from models.model import Model
from decimal import Decimal


class als_recommendations_securityid(Model):

    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        # FEATURE ENG
        date_to = datetime.today()
        date_from = date_to - relativedelta(days=self.config['feature_engineering']['timeframe_days'])
        logging.info(f"date_from: {date_from.strftime('%Y-%m-%d')} --> date_to: {date_to.strftime('%Y-%m-%d')}")
        
        self.query = utils.read_file_from_bucket(
            self.config['bucket_models'],
            self.config['feature_engineering']['s3_query_key']).replace(
            "{{date_from}}",
            "'" +
            date_from.strftime('%Y-%m-%d') +
            "'").replace(
            "{{date_to}}",
            "'" +
            date_to.strftime('%Y-%m-%d') +
            "'")
            
        self.dataset_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/dataset.csv'
        self.als_input_spark_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/als_input.csv'
        self.als_output_bucket = os.environ.get('ALS_OUTPUT_BUCKET')
        self.als_output_s3_key = f'{self.config["model_name"]}/{self.config["model_id"]}/output'
        self.als_metrics_s3_key = f'{self.config["model_name"]}/{self.config["model_id"]}/metrics'
        self.als_output_s3_uri = f's3://{self.als_output_bucket}/{self.als_output_s3_key}'
        self.als_metrics_s3_uri = f's3://{self.als_output_bucket}/{self.als_metrics_s3_key}'

    def data_preparation(self, df):
        logging.info("#############DATA PREPARATION#############")

        df_new, df_spark = self.custom_transformations(df)

        logging.info(f'Transformed DF: {df_new}')

        encoding = self.config['feature_engineering']['output_s3_encoding'] if 'output_s3_encoding' in self.config['feature_engineering'] else 'utf-8'

        # saves dataset
        upload = utils.put_obj_in_bucket(
            df_new,
            encoding,
            self.config['bucket_models'],
            self.dataset_key,
            'csv',
            True)
        logging.info(f'Putting df obj: {upload}')

        upload = utils.put_obj_in_bucket(
            df_spark,
            encoding,
            self.config['bucket_models'],
            self.als_input_spark_key,
            'csv',
            False)
        logging.info(f'Putting df obj: {upload}')

        output = {
            "s3_dataset_file": "s3://{}/{}".format(self.config['bucket_models'], self.dataset_key),
            "s3_als_input": "s3://{}/{}".format(self.config['bucket_models'], self.als_input_spark_key),
            "end_date": str(datetime.now())
        }

        return output

    def custom_transformations(self, df):
        logging.info("#############CUSTOM TRANSFORMATIONS ALS#############")
        df_new = df[~df.ticker.isin(['PESOS', 'DOLAR'])].copy()
        df_new.drop('ticker', axis=1, inplace=True)
        df_new.dropna(inplace=True)
        df_new.reset_index(inplace=True, drop=True)
        
        securityid = df_new.securityid.unique().reshape(-1, 1)
        encoder = OrdinalEncoder()
        encoder.fit(securityid)
        df_new['security_id_num'] = encoder.transform(np.array(df_new.securityid).reshape(-1, 1))
        df_new['security_id_num'] = df_new['security_id_num'].apply(np.int64)
        
        df_spark = df_new.drop(['grupo','securityid'], axis=1)

        return df_new, df_spark
        
        
    def train(self):
        step = "train"
        logging.info("#############STARTING MODEL TRAINING#############")
        sagemaker_session = sagemaker.Session()
        
        # Training model
        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()
        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"
        framework_version = self.config[step]["framework_version"] if 'framework_version' in self.config[step] else "3.0"
        max_runtime_in_seconds = self.config[step]["max_runtime_in_seconds"] if 'max_runtime_in_seconds' in self.config[step] else 1200
        

        print("framework_version: ",framework_version," , ",
        "sagemaker_session: ",sagemaker_session," , ",
        "base_job_name: ",self.config["model_id"]," , ",
        "role: ",self.config['role']," , ",
        "instance_count: ",instance_count," , ",
        "instance_type: ",instance_type," , "
        "output_kms_key: ",self.config['s3_kms_id']," , "
        "max_runtime_in_seconds: ",max_runtime_in_seconds," , "
        "network_config: ",NetworkConfig(security_group_ids=security_group_ids, subnets=subnet_ids)," , "
        "tags: ",tags)
        
        processor = PySparkProcessor(
            framework_version=framework_version,
            sagemaker_session=sagemaker_session,
            base_job_name=self.config["model_id"],
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            output_kms_key=self.config['s3_kms_id'],
            max_runtime_in_seconds=max_runtime_in_seconds,
            network_config=NetworkConfig(security_group_ids=security_group_ids, subnets=subnet_ids),
            tags=tags
        )
        
        # Upload python PySparkProcessor script to s3
        processing_job_script_s3_key = f"batch-inferences/{self.config['model_name']}/{self.config['model_id']}/processing_jobs/train_spark_als.py"
        processing_job_script_s3_uri = f"s3://{self.config['bucket_models']}/{processing_job_script_s3_key}"
        s3 = boto3.resource('s3')
        model_bucket = s3.Bucket(self.config['bucket_models']).upload_file('processing_jobs/train_spark_als.py', processing_job_script_s3_key)
        
        # Hyperparameters
        
        maxIter = self.config[step]["hyperparameters"]['maxIter'] if 'maxIter' in self.config[step]["hyperparameters"] else 10
        regParam = self.config[step]["hyperparameters"]['regParam'] if 'regParam' in self.config[step]["hyperparameters"] else 1
        alpha = self.config[step]["hyperparameters"]['alpha'] if 'alpha' in self.config[step]["hyperparameters"] else 1
        topN = self.config[step]["hyperparameters"]['topN'] if 'topN' in self.config[step]["hyperparameters"] else 10

        # Script Arguments
        arguments = [
            '--s3_input_data', self.config['dynamodb_item']['steps']['feature_engineering']['s3_als_input'],
            '--s3_output_data', self.als_output_s3_uri,
            '--s3_metrics_data', self.als_metrics_s3_uri,
            '--maxIter', str(maxIter), 
            '--regParam', str(regParam),
            '--alpha', str(alpha),
            '--topN', str(topN)
        ]
        print("processing_job_script_s3_uri: ", processing_job_script_s3_uri,"arguments: ",arguments)  
        processor.run(
            submit_app=processing_job_script_s3_uri,
            arguments=arguments,
            logs=False,
            wait=True
        )
        
        s3 = boto3.resource('s3')
        als_bucket = s3.Bucket(self.als_output_bucket)
        for object_summary in als_bucket.objects.filter(Prefix=self.als_output_s3_key):
            if object_summary.key[-3:] == 'csv':
                s3_als_output_uri = 's3://{}/{}'.format(als_bucket.name, object_summary.key)
                
        for object_summary in als_bucket.objects.filter(Prefix=self.als_metrics_s3_key):
            if object_summary.key[-3:] == 'csv':
                s3_als_metrics_uri = 's3://{}/{}'.format(als_bucket.name, object_summary.key)
                df_metrics = wr.s3.read_csv(s3_als_metrics_uri)
        
        output = {
            "end_date": str(datetime.now()),
            "hyperparameters": {
                "maxIter": str(df_metrics['maxIter'].iloc[0]), 
                "regParam": str(df_metrics['regParam'].iloc[0]),
                "topN": str(df_metrics['rank'].iloc[0]),
                "alpha": str(df_metrics['alpha'].iloc[0])
            },
            "metrics": {
                "rmse": str(df_metrics['RMSE'].iloc[0])
            },
            "s3_als_output": s3_als_output_uri
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def tuner(self):
        logging.info(
            "Tuner not available for als_recommendations_securityid")
        # TODO: https://towardsdatascience.com/build-recommendation-system-with-pyspark-using-alternating-least-squares-als-matrix-factorisation-ebe1ad2e7679
        # Hyperparameter tuning and cross validation
        step = "tuner"
        logging.info("#############STARTING MODEL TUNING#############")
        sagemaker_session = sagemaker.Session()
        
        # Training model
        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()
        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"
        framework_version = self.config[step]["framework_version"] if 'framework_version' in self.config[step] else "3.0"
        max_runtime_in_seconds = self.config[step]["max_runtime_in_seconds"] if 'max_runtime_in_seconds' in self.config[step] else 1200
        
        processor = PySparkProcessor(
            framework_version=framework_version,
            sagemaker_session=sagemaker_session,
            base_job_name=self.config["model_id"],
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            output_kms_key=self.config['s3_kms_id'],
            max_runtime_in_seconds=max_runtime_in_seconds,
            network_config=NetworkConfig(security_group_ids=security_group_ids, subnets=subnet_ids),
            tags=tags
        )
        
        # Upload python PySparkProcessor script to s3
        processing_job_script_s3_key = f"batch-inferences/{self.config['model_name']}/{self.config['model_id']}/processing_jobs/tuner_spark_als.py"
        processing_job_script_s3_uri = f"s3://{self.config['bucket_models']}/{processing_job_script_s3_key}"
        s3 = boto3.resource('s3')
        model_bucket = s3.Bucket(self.config['bucket_models']).upload_file('processing_jobs/tuner_spark_als.py', processing_job_script_s3_key)
        
        # Hyperparameters
        
        maxIter = self.config[step]["hyperparameters"]['maxIter'] if 'maxIter' in self.config[step]["hyperparameters"] else "10, 15"
        regParam = self.config[step]["hyperparameters"]['regParam'] if 'regParam' in self.config[step]["hyperparameters"] else "1, 0.5, 0.25, 0.1"
        alpha = self.config[step]["hyperparameters"]['alpha'] if 'alpha' in self.config[step]["hyperparameters"] else "1, 2, 4"
        topN = self.config[step]["hyperparameters"]['topN'] if 'topN' in self.config[step]["hyperparameters"] else 10

        # Script Arguments
        arguments = [
            '--s3_input_data', self.config['dynamodb_item']['steps']['feature_engineering']['s3_als_input'],
            '--s3_output_data', self.als_output_s3_uri,
            '--s3_metrics_data', self.als_metrics_s3_uri,
            '--maxIter', str(maxIter), 
            '--regParam', str(regParam),
            '--alpha', str(alpha),
            '--topN', str(topN)
        ]
        
        processor.run(
            submit_app=processing_job_script_s3_uri,
            arguments=arguments,
            logs=False,
            wait=True
        )
        
        s3 = boto3.resource('s3')
        als_bucket = s3.Bucket(self.als_output_bucket)
        for object_summary in als_bucket.objects.filter(Prefix=self.als_output_s3_key):
            if object_summary.key[-3:] == 'csv':
                s3_als_output_uri = 's3://{}/{}'.format(als_bucket.name, object_summary.key)
                
        for object_summary in als_bucket.objects.filter(Prefix=self.als_metrics_s3_key):
            if object_summary.key[-3:] == 'csv':
                s3_als_metrics_uri = 's3://{}/{}'.format(als_bucket.name, object_summary.key)
                df_metrics = wr.s3.read_csv(s3_als_metrics_uri)
                
        output = {
            "end_date": str(datetime.now()),
            "hyperparameters": {
                "maxIter": str(df_metrics['maxIter'].iloc[0]), 
                "regParam": str(df_metrics['regParam'].iloc[0]),
                "topN": str(df_metrics['rank'].iloc[0]),
                "alpha": str(df_metrics['alpha'].iloc[0])
            },
            "metrics": {
                "rmse": str(df_metrics['RMSE'].iloc[0])
            },
            "s3_als_output": s3_als_output_uri
        }

        self.config["dynamodb_item"]["steps"]["train"] = output

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
        logging.info(
            "Deploy not available for als_recommendations_securityid")
    
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