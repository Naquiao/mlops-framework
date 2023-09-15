import libs.utils as utils
import logging
import pandas as pd
import numpy as np
import sagemaker
import s3fs
import mxnet as mx
import os
import boto3
import tarfile
import awswrangler as wr
import itertools
from datetime import datetime
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from models.model import Model


class top_recommendations_securityid(Model):

    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        # FEATURE ENG
        self.fm_model_dynamo_table = os.environ.get('FM_MODEL_DYNAMO_TABLE')
        if self.fm_model_dynamo_table is None:
            raise Exception(
                "top_recommendations_securityid was expecting FM_MODEL_DYNAMO_TABLE as an environment variable and did not receive it")
        self.fm_dynamodb_item = utils.get_dynamodb_item(
            {"model_id": self.config['model_id'], "dynamodb_table": self.fm_model_dynamo_table})

        self.config['model_id']  = "knn-top-" + self.config['model_id']
        self.train_key           = f'data/{self.config["model_name"]}/{self.config["model_id"]}/train/train.protobuf'
        self.inference_file_name = 'inference.protobuf'
        self.inference_key       = f'batch-inferences/{self.config["model_name"]}/{self.config["model_id"]}/input/{self.inference_file_name}'
        self.results_key         = f'batch-inferences/{self.config["model_name"]}/{self.config["model_id"]}/results/results.csv'
        
        # DEPLOY
        self.predictor_cls = None
        # MONITORING
        self.baseline_dataset_format = DatasetFormat.csv(header=True)
        self.data_quality_s3_preprocessor_uri = None
        self.data_quality_s3_postprocessor_uri = None
        self.model_quality_s3_preprocessor_uri = None
        self.model_quality_s3_postprocessor_uri = None

    def feature_engineering(self):
        step = "feature_engineering"
        logging.info("#############STARTING FEATURE ENGINEERING#############")

        fm_s3_model_uri = self.fm_dynamodb_item["steps"]["train"]["s3_model_data"]
        fm_s3_model_uri_key = fm_s3_model_uri.replace(
            f"s3://{self.config['bucket_models']}/", '')

        logging.info(f'fm_s3_model_uri_key: {fm_s3_model_uri_key}')

        s3 = boto3.client('s3')
        s3.download_file(
            self.config['bucket_models'],
            fm_s3_model_uri_key,
            "model.tar.gz")
        tar = tarfile.open("model.tar.gz", 'r:gz')
        tar.extractall(path=".")
        os.system("unzip -o model_algo-1")
        os.system("mv symbol.json model-symbol.json")
        os.system("mv params model-0000.params")

        output = self.data_preparation()

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def data_preparation(self):
        logging.info("#############DATA PREPARATION#############")
        nb_users = int(
            self.fm_dynamodb_item["steps"]["feature_engineering"]["nb_users"])
        nb_items = int(
            self.fm_dynamodb_item["steps"]["feature_engineering"]["nb_items"])

        # Extract model data
        m = mx.module.Module.load(
            './model', 0, False, label_names=['out_label'])
        V = m._arg_params['v'].asnumpy()
        w = m._arg_params['w1_weight'].asnumpy()
        b = m._arg_params['w0_weight'].asnumpy()

        # item latent matrix - concat(V[i], w[i]).
        knn_item_matrix = np.concatenate((V[nb_users:], w[nb_users:]), axis=1)
        knn_train_label = np.arange(1, nb_items + 1)

        # user latent matrix - concat (V[u], 1)
        ones = np.ones(nb_users).reshape((nb_users, 1))
        knn_user_matrix = np.concatenate((V[:nb_users], ones), axis=1)

        encoding = self.config['feature_engineering']['output_s3_encoding'] if 'output_s3_encoding' in self.config['feature_engineering'] else 'utf-8'

        # saves training obj
        upload = utils.put_obj_in_bucket(
            (knn_item_matrix, knn_train_label),
            encoding,
            self.config['bucket_models'],
            self.train_key,
            self.config['feature_engineering']['dataset_format'])
        logging.info(f'Putting train obj: {upload}')
        
        # saves inference obj
        upload = utils.put_obj_in_bucket(
            (knn_user_matrix, None),
            encoding,
            self.config['bucket_models'],
            self.inference_key,
            self.config['feature_engineering']['dataset_format'])
        logging.info(f'Putting inference obj: {upload}')

        output = {
            "s3_train_file": "s3://{}/{}".format(self.config['bucket_models'], self.train_key),
            "s3_inference_file": "s3://{}/{}".format(self.config['bucket_models'], self.inference_key),
            "knn_train_feature_shape_x": knn_item_matrix.shape[0],
            "knn_train_feature_shape_y": knn_item_matrix.shape[1],
            "end_date": str(datetime.now())
        }

        return output

    def get_train_hyperparameters(self):
        h = self.config['train']['hyperparameters']

        hyperparameters = {
            "feature_dim": self.config["dynamodb_item"]["steps"]["feature_engineering"]["knn_train_feature_shape_y"]}

        for key in h.keys():
            hyperparameters[key] = h[key]

        return hyperparameters

    def get_tuner_hyperparameters(self):
        static_h = self.config['tuner']['hyperparameters']
        tuned_h = self.config['tuner']['tuned_hyperparameters']

        continuous_parameter_type = []

        integer_parameter_type = [
            "k",
            "sample_size"
        ]

        static_hyperparameters = {
            "feature_dim": self.config["dynamodb_item"]["steps"]["feature_engineering"]["knn_train_feature_shape_y"]}

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
    
    def deploy(self):
        step = "deploy"
        logging.info("#############STARTING BATCH TRANSFORM JOB#############")

        training_job_name      = self.config["dynamodb_item"]["steps"]["train"]["job_name"]
        initial_instance_count = self.config[step]["initial_instance_count"] if 'initial_instance_count' in self.config[step] else 1
        instance_type          = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.m5.large"
        max_payload            = self.config[step]["max_payload"] if 'max_payload' in self.config[step] else 1
        strategy               = self.config[step]["strategy"] if 'strategy' in self.config[step] else None
        assemble_with          = self.config[step]["assemble_with"] if 'assemble_with' in self.config[step] else None
        model_name             = self.config["model_id"][:63]
        s3_uri_input_file      = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_inference_file"]
        s3_output_path         = f"s3://{self.config['bucket_models']}/batch-inferences/{self.config['model_name']}/{self.config['model_id']}/output/"
        
        tags = self.get_model_tags()

        attached_estimator = sagemaker.estimator.Estimator.attach(
            training_job_name)
        transformer = attached_estimator.transformer(
            instance_count=initial_instance_count,
            instance_type=instance_type,
            max_payload=1,
            strategy=strategy,
            assemble_with=assemble_with,
            model_name=model_name,
            output_path=s3_output_path,
            output_kms_key=self.config['s3_kms_id'],
            accept="application/jsonlines; verbose=true",
            tags=tags
        )

        logging.info("#############RUNNING TRANSFORMER#############")
        transformer.transform(data = s3_uri_input_file,
                        content_type="application/x-recordio-protobuf",
                        split_type='RecordIO',
                        wait=True,
                        logs=False
                        )
        
        job_name = transformer.latest_transform_job.name
        logging.info(f'transform job status: {transformer.sagemaker_session.describe_transform_job(job_name)}')
        job_status = transformer.sagemaker_session.describe_transform_job(job_name)["TransformJobStatus"]

        output = {
            "model_name": model_name,
            "batch_job_name": job_name,
            "batch_job_status": job_status,
            "instance_type": instance_type,
            "initial_instance_count": initial_instance_count,
            "s3_uri_input_file": s3_uri_input_file,
            "s3_uri_output_file": s3_output_path + self.inference_file_name + ".out",
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])
        
        if job_status == 'Failed':
            raise Exception(f"Batch job {job_name} failed. {output}")
        
    def post_processing(self):
        step = "post_processing"
        logging.info("#############STARTING POST PROCESSING#############")
        '''
        This function will recreate the original dataset with the top k recommendations for user and securityid.
        Dataset format:       
        idcuenta       grupo           securityid  distances
        0          1000  Renta Fija   AL30-0002-C-CT-ARS   0.669939
        1          1000     CEDEARS    CAT-0001-C-CT-ARS   0.685302
        2          1000  Renta Fija   SG0X-0003-C-CT-ARS   0.685343
        3          1000    Acciones   CRES-0001-C-CT-ARS   0.717752
        4          1000  Renta Fija  GD30D-0001-C-CT-USD   0.773322
        ...         ...         ...                  ...        ...
        432725   158148  Renta Fija  GD30D-0001-C-CT-USD   0.817277
        432726   158148     CEDEARS    AUY-0001-C-CT-ARS   0.897473
        432727   158148  Renta Fija  IRC5O-0003-C-CT-ARS   1.145357
        432728   158148  Renta Fija  AL30D-0002-C-CT-USD   2.015768
        432729   158148  Renta Fija   AL30-0002-C-CT-ARS   3.161978

        Where the lower distance means a stronger recommendation.
        '''
        
        # dataset original de s3
        dataset = wr.s3.read_csv(self.fm_dynamodb_item["steps"]["feature_engineering"]["s3_dataset_file"])
        
        # diccionario (posicion , usuario_securityid)
        matrix_index_key = self.fm_dynamodb_item["steps"]["feature_engineering"]["s3_matrix_index_file"].replace(f"s3://{self.config['bucket_models']}/",'')
        original_matrix = utils.get_dict_from_s3_json(
            self.config['bucket_models'], matrix_index_key)
        
        # inferencias top k para cada usuario
        inference_output_file = self.config["dynamodb_item"]["steps"]["deploy"]["s3_uri_output_file"]
        df_inference_output   = wr.s3.read_json(inference_output_file, lines=True)
        
        # cantidad de idcuenta y cantidad de securityid
        nb_users = int(self.fm_dynamodb_item["steps"]["feature_engineering"]["nb_users"])
        nb_items = int(self.fm_dynamodb_item["steps"]["feature_engineering"]["nb_items"])
        logging.info(f'nb_users+nb_items={nb_users+nb_items}')
        logging.info(f'length original_matrix: {len(original_matrix)}')
        
        # k longitud de las inferencias por usuario
        k = len(df_inference_output['distances'][0]) 

        # Distancia serie con las distancias de las inferencias concatenadas para todos los usuarios obtenidad de df_inference_output
        distances = pd.Series(list(itertools.chain(*df_inference_output['distances'])),name='distances')
        # labels serie con los index de la matrix_original correspondientes al securityid concatenadas para todas las inferencias obtenidad de df_inference_output
        labels = pd.Series(np.array(list(itertools.chain(*df_inference_output['labels'])))+nb_users,name='label')
        # dataframe identificando securityid-grupo
        grupo = dataset[['grupo','securityid']].drop_duplicates()
        
        # original_matrix como dataframe con orden de la columna y etiqueta del idcuenta o securityid
        original_matrix2 = pd.DataFrame({'order' : list(original_matrix.keys()),'idcuenta':list(original_matrix.values())})

        # dataframe con orden de la columna y etiqueta del idcuenta
        usuarios = original_matrix2.iloc[0:nb_users,:]
        # dataframe con orden de la columna y etiqueta del securityid
        labels2 = original_matrix2.iloc[nb_users:,:].rename(columns={'order':'label','idcuenta':'securityid'})
        labels2.label = pd.to_numeric(labels2.label)
        
        # replicar k veces a cada usuario
        df = list(usuarios.order)*k
        df = pd.Series(df, name = 'order').sort_values().reset_index().merge(usuarios)
        # concatena la distancia y el label de las inferencias, que representa el número de columna correspondiente al securiyid 
        df = pd.concat([df,labels,distances],axis=1)
        
        # a partir de la columna label hacemos el merge para obtener el label del securityid
        df = df.merge(labels2,on='label',how='left')
        # eliminamos columnas innecesarias
        df.drop(['index','order','label'],axis=1,inplace=True)
        # agregamos el grupo correspondiente a cada securityid
        df = df.merge(grupo,on='securityid',how='left')
        df = df[['idcuenta','grupo','securityid','distances']]
        
        logging.info(df)
        
        header = True
        # saves new dataset
        upload = utils.put_obj_in_bucket(
            df,
            'utf-8',
            self.config['bucket_models'],
            self.results_key,
            'csv',
            header)
        logging.info(f'Putting train obj: {upload}')
        
        s3_results_uri = f"s3://{self.config['bucket_models']}/{self.results_key}"
        
        output = {
            "model_name": self.config["model_name"],
            "model_id": self.config["model_id"],
            "s3_results_uri": s3_results_uri,
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])
        
        message = {
                "Model Name": self.config["model_name"],
                "Model ID": self.config['model_id'],
                "S3 Results URI": s3_results_uri
            }
        utils.send_sns_notification(
            self.config['sns_topic_arn'], message)