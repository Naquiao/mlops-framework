import libs.utils as utils
import logging
import pandas as pd
import numpy as np
import sagemaker
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from models.model import Model


class recommendations_securityid(Model):

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
        logging.info("#############DATA PREPARATION#############")

        df_new = self.custom_transformations(df)

        logging.info(f'Transformed DF: {df_new}')

        train_obj, test_obj, column_dict, nb_users, nb_items = self.cut_df(
            df_new)

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

        # saves matrix index
        upload = utils.put_obj_in_bucket(
            column_dict,
            encoding,
            self.config['bucket_models'],
            self.matrix_index_key,
            'json')
        logging.info(f'Putting matrix_index obj: {upload}')

        # saves training matrix
        upload = utils.put_obj_in_bucket(
            train_obj,
            encoding,
            self.config['bucket_models'],
            self.train_key,
            self.config['feature_engineering']['dataset_format'])
        logging.info(f'Putting train obj: {upload}')

        # saves testing matrix
        upload = utils.put_obj_in_bucket(
            test_obj,
            encoding,
            self.config['bucket_models'],
            self.test_key,
            self.config['feature_engineering']['dataset_format'])
        logging.info(f'Putting test obj: {upload}')

        output = {
            "s3_dataset_file": "s3://{}/{}".format(self.config['bucket_models'], self.dataset_key),
            "s3_matrix_index_file": "s3://{}/{}".format(self.config['bucket_models'], self.matrix_index_key),
            "s3_train_file": "s3://{}/{}".format(self.config['bucket_models'], self.train_key),
            "s3_test_file": "s3://{}/{}".format(self.config['bucket_models'], self.test_key),
            "nb_users": nb_users,
            "nb_items": nb_items,
            "end_date": str(datetime.now())
        }

        return output

    def custom_transformations(self, df):
        logging.info("#############CUSTOM TRANSFORMATIONS#############")
        df_new = df[~df.ticker.isin(['PESOS', 'DOLAR'])].copy()
        df_new.drop('ticker', axis=1, inplace=True)
        df_new.dropna(inplace=True)
        df_new.reset_index(inplace=True, drop=True)

        return df_new

    def cut_df(self, df):
        logging.info("#############CUT DF#############")

        # identifico a los usuarios con mas de una inversion
        user_group = df.groupby('idcuenta').q.count().sort_values()
        users_varias_inv = user_group.index[df.groupby(
            'idcuenta').q.count().sort_values() > 1]

        # Ultima inversion para cada usuario
        user_group_2 = df.loc[df['idcuenta'].isin(
            users_varias_inv)].reset_index().groupby('idcuenta').nth(-1)

        index_test = user_group_2['index'].values
        index_train = [elem for elem in df.index if elem not in index_test]

        enc = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
        enc.fit(df.iloc[index_train, [0, 2]])

        column_dict = {}
        i = 0
        for user in enc.categories_[0]:
            column_dict[i] = str(user)
            i += 1

        nb_users = len(enc.categories_[0])

        for item in enc.categories_[1]:
            column_dict[i] = str(item)
            i += 1

        nb_items = len(enc.categories_[1])

        logging.info("x_train")
        X_train = enc.transform(df.iloc[index_train, [0, 2]]).astype('float32')
        logging.info("y_train")
        Y_train = df.q.iloc[index_train].values.astype('float32')
        logging.info("x_test")
        X_test = enc.transform(df.iloc[index_test, [0, 2]]).astype('float32')
        logging.info("y_test")
        Y_test = df.q.iloc[index_test].values.astype('float32')

        return (X_train, Y_train), (X_test,
                                    Y_test), column_dict, nb_users, nb_items

    def get_train_hyperparameters(self):
        h = self.config['train']['hyperparameters']

        matrix_index = utils.get_dict_from_s3_json(
            self.config['bucket_models'], self.matrix_index_key)
        feature_dim = len(matrix_index)
        logging.info(f'feature_dim: {feature_dim}')

        hyperparameters = {
            "feature_dim": feature_dim,
        }

        for key in h.keys():
            hyperparameters[key] = h[key]

        return hyperparameters

    def get_tuner_hyperparameters(self):
        static_h = self.config['tuner']['hyperparameters']
        tuned_h = self.config['tuner']['tuned_hyperparameters']

        continuous_parameter_type = [
            "bias_init_scale",
            "bias_init_sigma",
            "bias_init_value",
            "bias_lr",
            "bias_wd",
            "factors_init_scale",
            "factors_init_sigma",
            "factors_init_value",
            "factors_lr",
            "factors_wd",
            "linear_init_scale",
            "linear_init_sigma",
            "linear_init_value",
            "linear_lr",
            "linear_wd"
        ]

        integer_parameter_type = [
            "epoch",
            "mini_batch_size"
        ]

        matrix_index = utils.get_dict_from_s3_json(
            self.config['bucket_models'], self.matrix_index_key)
        feature_dim = len(matrix_index)
        logging.info(f'feature_dim: {feature_dim}')

        static_hyperparameters = {
            "feature_dim": feature_dim,
        }

        tuned_hyperparameters = {}

        for key in static_h.keys():
            static_hyperparameters[key] = static_h[key]

        for key in tuned_h.keys():
            if key in continuous_parameter_type:
                tuned_hyperparameters[key] = ContinuousParameter(
                    tuned_h[key][0], tuned_h[key][1])
            elif key in integer_parameter_type:
                tuned_hyperparameters[key] = IntegerParameter(
                    tuned_h[key][0], tuned_h[key][1])

        return static_hyperparameters, tuned_hyperparameters

    def validate(self):
        step = "validate"
        metric_objective = self.config[step]['metric']
        optimization_direction = self.config[step]['optimization_direction']
        base_value = self.config[step]['base_value']
        current_objective_value = self.config["dynamodb_item"]["steps"]["train"]["metrics"][metric_objective]

        logging.info("#############STARTING VALIDATION#############")

        logging.info(
            f'metric: {metric_objective}, direction: {optimization_direction}, base_value: {base_value}')
        logging.info(f'current_objective_value: {current_objective_value}')

        validation_details = {
            metric_objective + "_base": base_value,
            metric_objective + "_current": current_objective_value
        }

        validation_success = False

        if self.compare_optimization_values(
                current_objective_value,
                base_value,
                optimization_direction):
            logging.info(
                f'Validation success')
            validation_success = True
        else:
            logging.info(
                f'Validation failed')
            validation_success = False

        output = {
            "validation_success": validation_success,
            "validation_details": validation_details,
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

        if not validation_success:
            raise Exception(
                f'Validation failed, retrain with different hyperparameters or change the base_value set in config file: {validation_details}')

    def post_processing(self):
        step = "post_processing"
        logging.info(
            "#############STARTING KNN TOP RECOMMENDATIONS SECURITY ID PIPELINE#############")

        knn_task_definition_arn = utils.get_fargate_task_definition_arn(
            self.config[step]['knn_container_name'])

        logging.info(f'knn_task_definition_arn: {knn_task_definition_arn}')

        input_json = {
            "task_definition_arn": knn_task_definition_arn,
            "container_name": self.config[step]["knn_container_name"],
            "model_id": self.config['model_id']
        }

        logging.info(f'Input: {input_json}')

        timename = datetime.now().strftime("_%d%m%Y%H%M%S")
        sfn_name = "top-" + input_json["model_id"]
        sfn_name = sfn_name[:80 - len(timename)] + timename

        logging.info(
            "Starting StepFunction for {}".format(
                input_json["task_definition_arn"]))

        sfn_response = utils.start_stepfunction(
            self.config[step]['sfn_fargate_arn'], sfn_name, input_json)

        logging.info(f'Launched StepFunction: {sfn_response}')

        output = {
            "sfn_name": sfn_name,
            "sfn_execution_arn": sfn_response['executionArn'],
            "task_definition_arn": knn_task_definition_arn,
            "container_name": self.config[step]["knn_container_name"],
            "model_id": self.config['model_id'],
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])
