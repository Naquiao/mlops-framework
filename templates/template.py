import libs.utils as utils
import logging
import pandas as pd
import numpy as np
import sagemaker
from datetime import datetime
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from models.model import Model


class template(Model):

    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        # FEATURE ENG
        self.query = utils.read_file_from_bucket(self.config['bucket_models'], self.config['feature_engineering']['s3_query_key']).replace(
            "{{date_from}}", "'" + self.config["feature_engineering"]["query_date_from"] + "'").replace(
            "{{date_to}}", "'" + self.config["feature_engineering"]["query_date_to"] + "'")
        self.dataset_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/dataset.csv'
        self.matrix_index_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/dataset/matrix_index.json'
        self.train_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/train/train.protobuf'
        self.test_key = f'data/{self.config["model_name"]}/{self.config["model_id"]}/test/test.protobuf'
        # DEPLOY
        self.predictor_cls = None 
        # MONITORING
        self.baseline_dataset_format = DatasetFormat.csv(header=True)
        self.data_quality_s3_preprocessor_uri = None
        self.data_quality_s3_postprocessor_uri = None
        self.model_quality_s3_preprocessor_uri = None
        self.model_quality_s3_postprocessor_uri = None

    def data_preparation(self, df):
        '''
        This function will receive the result of the query made in a dataframe (df)
        Process the dataframe, make custom transformations, format the dataset following the requirements of your algorithm.
        Split the dataset for testing, training and validation if needed.
        Create new files if needed.
        Upload them to s3 (use utils.put_obj_in_bucket function)
        Return the location of the files uploaded in a dictionary to be used by the training step afterwards.
        This output will be saves in the dynamodb object.
        e.g.
        dynamodb_output = {
           "s3_dataset_file": "s3://{}/{}".format(self.config['bucket_models'], self.dataset_key),
           "s3_matrix_index_file": "s3://{}/{}".format(self.config['bucket_models'], self.matrix_index_key),
           "s3_train_file": "s3://{}/{}".format(self.config['bucket_models'], self.train_key),
           "s3_test_file": "s3://{}/{}".format(self.config['bucket_models'], self.test_key),
           "end_date": str(datetime.now())
        }
        You can set up the variables needed in the function by setting them up in the config file and get them using the self.config variable.
        '''
        dynamodb_output = {}
        return dynamodb_output

    def get_train_hyperparameters(self):
        '''
        This function should return the hyperparameters that will be used by the algorithm when running a training job
        This will run if enable_tuner = false in config
        e.g.
        hyperparameters = {
            "hyperparameter_1": hyperparametervalue_1,
            "hyperparameter_2": hyperparametervalue_2,
            "hyperparameter_3": hyperparametervalue_3
        }
        You can set up the variables needed in the function by setting them up in the config file and get them using the self.config variable.
        '''
        hyperparameters = {}        
        return hyperparameters

    def get_tuner_hyperparameters(self):
        '''
        This function should return the static and tuned hyperparameters that will be used by the algorithm when running an hyperparameter tunning job
        This will run if enable_tuner = true in config
        e.g.
        static_hyperparameters = {
            "static_hyperparameter_1": static_hyperparameter_value_1,
            "static_hyperparameter_2": static_hyperparameter_value_2,
            "static_hyperparameter_3": static_hyperparameter_value_3
        }
        tuned_hyperparameters = {
            "tuned_hyperparameter_1": ContinuousParameter(tuned_hyperparameter_1_min_range, tuned_hyperparameter_1_max_range),
            "tuned_hyperparameter_2": ContinuousParameter(tuned_hyperparameter_2_min_range, tuned_hyperparameter_2_max_range),
            "tuned_hyperparameter_3": IntegerParameter(tuned_hyperparameter_3_min_range, tuned_hyperparameter_3_max_range)
            
        }
        You can set up the variables needed in the function by setting them up in the config file and get them using the self.config variable.
        '''
        static_hyperparameters = {}
        tuned_hyperparameters = {}
        return static_hyperparameters, tuned_hyperparameters

    def validate(self):
        '''
        This function is used to perform validations to the trained model if needed.
        No return value is needed, but if you want to stop the pipeline if the validation failed your expectations you can raise an Exception
        One example if validation could be if one of the metrics obtanied in the training job is less or greather than a fixed value
        This function will run if enable_validate = true in config
        You can set up the variables needed in the function by setting them up in the config file and get them using the self.config variable.
        '''
        pass
    
    def get_autoscaling_policies(self):
        '''
        This function is used to set up the autoscaling policy for the endpoint.
        This policy will be passed as a variable to parent class put_scaling_policy method, TargetTrackingScalingPolicyConfiguration parameter
        doc: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/application-autoscaling.html#ApplicationAutoScaling.Client.put_scaling_policy
        This function will run if deploy.enable_autoscaling = true in config
        '''
        policies = []
        return policies