from abc import ABC, abstractmethod
import logging
import libs.utils as utils
import pandas as pd
import boto3
import sagemaker
import json
from datetime import datetime
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.inputs import TrainingInput
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from decimal import Decimal


class Model(ABC):

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.date_format = "%Y-%m-%d-%H-%M-%S"

    def post_processing(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def data_preparation(self, df):
        pass

    def compare_optimization_values(
            self,
            current_objective_value,
            base_value,
            optimization_direction):
        if optimization_direction.lower() == "minimize":
            return True if current_objective_value < base_value else False
        elif optimization_direction.lower() == "maximize":
            return True if current_objective_value > base_value else False
        else:
            raise Exception("Invalid optimization direction")

    def get_model_subnet_ids(self):
        subnet_ids = self.config["subnet_ids"] if 'subnet_ids' in self.config else None
        return subnet_ids

    def get_security_group_ids(self):
        security_group_ids = self.config["security_group_ids"] if 'security_group_ids' in self.config else None
        return security_group_ids

    def get_model_tags(self):
        tags = self.config["tags"] if 'tags' in self.config else None
        return tags

    def feature_engineering(self):
        step = "feature_engineering"
        logging.info("#############STARTING FEATURE ENGINEERING#############")

        filename = utils.make_athena_query(self.query, self.config)
        results = utils.read_csv_from_s3(filename, self.config)
        df = pd.DataFrame(data=results)

        logging.info(f'DF: {df}')
        logging.info("Applying custom transformations")

        output = self.data_preparation(df)

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def train(self):
        step = "train"
        logging.info("#############STARTING MODEL TRAINING#############")
        sagemaker_session = sagemaker.Session()

        s3_output_path = f"s3://{self.config['bucket_models']}/{self.config[step]['output_prefix']}"

        image_uri = sagemaker.image_uris.retrieve(
            region=self.config["region"],
            framework=self.config[step]['algorithm'],
            version=self.config[step]['algorithm_version'])

        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()

        logging.info(f'subnet_ids: {subnet_ids}')
        logging.info(f'tags: {tags}')
        logging.info(f'security_group_ids: {security_group_ids}')

        # Training model
        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"

        estimator = sagemaker.estimator.Estimator(
            sagemaker_session=sagemaker_session,
            image_uri=image_uri,
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            base_job_name=self.config["model_id"],
            output_path=s3_output_path,
            output_kms_key=self.config['s3_kms_id'],
            subnets=subnet_ids,
            security_group_ids=security_group_ids,
            tags=tags
        )

        hyperparameters = self.get_train_hyperparameters()

        estimator.set_hyperparameters(**hyperparameters)

        # define the data type and paths to the training and validation
        # datasets
        content_type = self.config[step]['training_datasets_content_type'] if 'training_datasets_content_type' in self.config[step] else None

        data_channels = {}

        if 'data_channel_train_key' in self.config[step]:
            s3_input_train = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_train_file"]
            train_input = TrainingInput(
                s3_input_train, content_type=content_type)
            data_channels[self.config[step]
                          ['data_channel_train_key']] = train_input

        if 'data_channel_test_key' in self.config[step]:
            s3_input_test = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_test_file"]
            testing_input = TrainingInput(
                s3_input_test, content_type=content_type)
            data_channels[self.config[step]
                          ['data_channel_test_key']] = testing_input

        estimator.fit(inputs=data_channels, logs=False)
        #model.tar.gz is generated

        client = boto3.client('sagemaker')

        response = client.describe_training_job(
            TrainingJobName=estimator.latest_training_job.name
        )

        training_billable_time = response.get('BillableTimeInSeconds')

        logging.info(f'describe training job: {response}')

        logging.info(f'metrics list: {response["FinalMetricDataList"]}')

        metric_dict = {}

        # removing timestamp from FinalMetricDataList
        for metric in response["FinalMetricDataList"]:
            metric_dict[metric["MetricName"]] = Decimal(metric["Value"])

        logging.info(f'metric_dict: {metric_dict}')

        output = {
            "image_uri": image_uri,
            "hyperparameters": hyperparameters,
            "job_name": estimator.latest_training_job.name,
            "s3_model_data": estimator.model_data,
            "end_date": str(datetime.now()),
            "metrics": metric_dict,
            "billable_time": training_billable_time
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def tuner(self):
        step = "tuner"

        logging.info("#############STARTING MODEL TUNING#############")
        sagemaker_session = sagemaker.Session()

        s3_output_path = f"s3://{self.config['bucket_models']}/{self.config[step]['output_prefix']}"

        image_uri = sagemaker.image_uris.retrieve(
            region=self.config["region"],
            framework=self.config[step]['algorithm'],
            version=self.config[step]['algorithm_version'])

        subnet_ids = self.get_model_subnet_ids()
        tags = self.get_model_tags()
        security_group_ids = self.get_security_group_ids()

        logging.info(f'subnet_ids: {subnet_ids}')
        logging.info(f'tags: {tags}')
        logging.info(f'security_group_ids: {security_group_ids}')

        instance_count = self.config[step]["instance_count"] if 'instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.c4.xlarge"

        estimator = sagemaker.estimator.Estimator(
            sagemaker_session=sagemaker_session,
            image_uri=image_uri,
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            base_job_name=self.config["model_id"],
            output_path=s3_output_path,
            output_kms_key=self.config['s3_kms_id'],
            subnets=subnet_ids,
            security_group_ids=security_group_ids,
            tags=tags
        )

        static_hyperparameters, tuned_hyperparameters = self.get_tuner_hyperparameters()

        estimator.set_hyperparameters(**static_hyperparameters)

        max_jobs = self.config[step]["max_jobs"] if 'max_jobs' in self.config[step] else 1
        max_parallel_jobs = self.config[step]["max_parallel_jobs"] if 'max_parallel_jobs' in self.config[step] else 1

        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name=self.config[step]['objective_metric_name'],
            objective_type=self.config[step]['optimization_direction'],
            hyperparameter_ranges=tuned_hyperparameters,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            base_tuning_job_name=self.config['model_id'],
            tags=tags)

        content_type = self.config[step]['training_datasets_content_type'] if 'training_datasets_content_type' in self.config[step] else None
        data_channels = {}

        if 'data_channel_train_key' in self.config[step]:
            s3_input_train = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_train_file"]
            train_input = TrainingInput(
                s3_input_train, content_type=content_type)
            data_channels[self.config[step]
                          ['data_channel_train_key']] = train_input

        if 'data_channel_test_key' in self.config[step]:
            s3_input_test = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_test_file"]
            testing_input = TrainingInput(
                s3_input_test, content_type=content_type)
            data_channels[self.config[step]
                          ['data_channel_test_key']] = testing_input

        tuner.fit(inputs=data_channels, logs=False)

        client = boto3.client('sagemaker')
        response = client.describe_training_job(
            TrainingJobName=tuner.best_training_job()
        )

        tuning_jobs_df = tuner.analytics().dataframe()
        tuning_billable_time = tuning_jobs_df['TrainingElapsedTimeSeconds'].sum()
        logging.info(f'describe training job: {response}')

        logging.info(f'metrics list: {response["FinalMetricDataList"]}')

        metric_dict = {}

        # removing timestamp from FinalMetricDataList
        for metric in response["FinalMetricDataList"]:
            metric_dict[metric["MetricName"]] = Decimal(metric["Value"])

        logging.info(f'metric_dict: {metric_dict}')

        output = {
            "image_uri": image_uri,
            "job_name": tuner.best_training_job(),
            "hyperparameters": tuner.best_estimator().hyperparameters(),
            "s3_model_data": tuner.best_estimator().model_data,
            "end_date": str(datetime.now()),
            "metrics": metric_dict,
            "billable_time" : tuning_billable_time
        }

        self.config["dynamodb_item"]["steps"]["train"] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def registry(self):
        step = "registry"
        logging.info("#############STARTING MODEL REGISTRY#############")

        image_uri = self.config["dynamodb_item"]["steps"]["train"]["image_uri"]
        content_types = self.config[step]['supported_content_types']
        response_types = self.config[step]['supported_response_mime_types']
        model_data_url = self.config["dynamodb_item"]["steps"]["train"]["s3_model_data"]
        model_package_group_name = self.config[step]['model_package_group_name']
        model_package_description = f'Model package for {self.config["model_id"]} made in {datetime.now()}'
        # 'Approved'|'Rejected'|'PendingManualApproval'
        model_approval_status = self.config[step]['model_approval_status']

        model_package_arn = self.register_model_version(
            image_uri,
            content_types,
            response_types,
            model_data_url,
            model_package_group_name,
            model_package_description,
            model_approval_status)

        output = {
            "model_package_group_name": model_package_group_name,
            "model_package_arn": model_package_arn,
            "end_date": str(datetime.now())
        }

        self.config["dynamodb_item"]["steps"][step] = output

        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def register_model_version(
            self,
            image_uri,
            content_types,
            response_types,
            model_data_url,
            model_package_group_name,
            model_package_description,
            model_approval_status):
        sm_client = boto3.client('sagemaker')

        modelpackage_inference_specification = {
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": image_uri,
                    }
                ],
                "SupportedContentTypes": content_types,
                "SupportedResponseMIMETypes": response_types,
            }
        }

        # Specify the model data
        modelpackage_inference_specification["InferenceSpecification"][
            "Containers"][0]["ModelDataUrl"] = model_data_url

        create_model_package_input_dict = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageDescription": model_package_description,
            "ModelApprovalStatus": model_approval_status
        }
        create_model_package_input_dict.update(
            modelpackage_inference_specification)

        create_mode_package_response = sm_client.create_model_package(
            **create_model_package_input_dict)
        model_package_arn = create_mode_package_response["ModelPackageArn"]
        logging.info('ModelPackage Version ARN : {}'.format(model_package_arn))
        return model_package_arn

    def get_data_capture_config(self):
        if 'data_capture_config' in self.config["deploy"]:
            sampling_percentage = self.config["deploy"]["data_capture_config"][
                "sampling_percentage"] if 'sampling_percentage' in self.config["deploy"]["data_capture_config"] else 20
            capture_options = self.config["deploy"]["data_capture_config"]["capture_options"] if 'capture_options' in self.config["deploy"]["data_capture_config"] else [
                "REQUEST", "RESPONSE"]
            s3_capture_upload_path = f"s3://{self.config['bucket_models']}/{self.config['deploy']['data_capture_config']['output_prefix']}"

            logging.info(
                f'Configuring DataCapture: sampling_percentage: {sampling_percentage} # capture_options: {capture_options} # s3_capture_upload_path: {s3_capture_upload_path}')

            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=sampling_percentage,
                destination_s3_uri=s3_capture_upload_path,
                kms_key_id=self.config['s3_kms_id'],
                capture_options=capture_options
            )
            return data_capture_config
        else:
            return None

    def set_autoscaling(self, endpoint_name):
        try:
            logging.info("#############APPLYING AUTOSCALING#############")
            client = boto3.client('application-autoscaling')

            autoscaling_output = {}

            response = client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=self.config["deploy"]["autoscaling"]["min_capacity"],
                MaxCapacity=self.config["deploy"]["autoscaling"]["max_capacity"])

            policies_list = []
            for policy in self.get_autoscaling_policies():
                response = client.put_scaling_policy(
                    PolicyName=f'{endpoint_name}-{policy["CustomizedMetricSpecification"]["MetricName"]}-scaling-policy',
                    ServiceNamespace='sagemaker',
                    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                    PolicyType='TargetTrackingScaling',
                    TargetTrackingScalingPolicyConfiguration=policy)
                logging.info(f'response: {response}')
                policy_output = {
                    "PolicyARN": response['PolicyARN'],
                    "Alarms": response['Alarms']
                }
                policies_list.append(policy_output)

            autoscaling_output["min_capacity"] = self.config["deploy"]["autoscaling"]["min_capacity"]
            autoscaling_output["max_capacity"] = self.config["deploy"]["autoscaling"]["max_capacity"]
            autoscaling_output["policies"] = policies_list

            return autoscaling_output
        except Exception as e:
            logging.info(f'set autoscaling error: {e}')

    def deploy(self):
        step = "deploy"
        logging.info("#############STARTING DEPLOY#############")

        training_job_name = self.config["dynamodb_item"]["steps"]["train"]["job_name"]
        initial_instance_count = self.config[step][
            "initial_instance_count"] if 'initial_instance_count' in self.config[step] else 1
        instance_type = self.config[step]["instance_type"] if 'instance_type' in self.config[step] else "ml.m5.large"
        model_name = self.config["model_id"][:63]
        endpoint_name = self.config["model_id"][:63]
        data_capture_config = self.get_data_capture_config()
        tags = self.get_model_tags()

        attached_estimator = sagemaker.estimator.Estimator.attach(
            training_job_name)
        attached_estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            predictor_cls=self.predictor_cls,
            model_name=model_name,
            endpoint_name=endpoint_name,
            data_capture_config=data_capture_config,
            tags=tags
        )

        autoscaling_output = {}
        if 'enable_autoscaling' in self.config[step] and self.config[step]['enable_autoscaling']:
            autoscaling_output = self.set_autoscaling(endpoint_name)

        output = {
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "instance_type": instance_type,
            "initial_instance_count": initial_instance_count,
            "end_date": str(datetime.now()),
            "autoscaling": autoscaling_output
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

    def delete(self):
        step = "delete"
        logging.info("#############STARTING DELETE#############")
        if 'is_production' in self.config["dynamodb_item"] and self.config["dynamodb_item"]["is_production"]:
            raise Exception(
                f'ModelId {self.config["model_id"]} tagged as production in dynamodb. Cant delete it')
        else:
            endpoint_name = self.config["dynamodb_item"]["steps"]["deploy"]["endpoint_name"]
            model_name = self.config["dynamodb_item"]["steps"]["deploy"]["model_name"]

            utils.sagemaker_delete_monitoring_schedules(endpoint_name)
            utils.sagemaker_delete_model(model_name)
            utils.sagemaker_delete_endpoint(endpoint_name)
            utils.sagemaker_delete_endpoint_config(endpoint_name)
            utils.cloudwatch_delete_dashboard(endpoint_name)

            output = {
                "end_date": str(datetime.now())
            }

            self.config["dynamodb_item"]["steps"][step] = output
            utils.update_dynamodb_item(
                self.config["dynamodb_table"],
                self.config["dynamodb_item"])

            message = {
                "Model Deleted": self.config['model_name'],
                "Model ID": self.config['model_id'],
                "Endpoint ID": endpoint_name
            }

            utils.send_sns_notification(
                self.config['sns_topic_arn'], message)

    def monitoring(self):
        step = "monitoring"
        logging.info("#############STARTING MONITORING#############")

        create_cloudwatch_dashboard = self.config[step][
            'create_cloudwatch_dashboard'] if 'create_cloudwatch_dashboard' in self.config[step] else True

        if create_cloudwatch_dashboard:
            self.create_cloudwatch_dashboard()

        if 'data_quality' in self.config[step] and 'enabled' in self.config[step][
                'data_quality'] and self.config[step]['data_quality']['enabled']:
            self.monitor_data_quality()
        if 'model_quality' in self.config[step] and 'enabled' in self.config[step][
                'model_quality'] and self.config[step]['model_quality']['enabled']:
            self.monitor_model_quality()
        if 'bias_drift' in self.config[step] and 'enabled' in self.config[
                step]['bias_drift'] and self.config[step]['bias_drift']['enabled']:
            self.monitor_bias_drift()
        if 'feature_attribution_drift' in self.config[step] and 'enabled' in self.config[step][
                'feature_attribution_drift'] and self.config[step]['feature_attribution_drift']['enabled']:
            self.monitor_feature_attribution_drift()

    def create_cloudwatch_dashboard(self):
        cw = boto3.client('cloudwatch')

        dashboard_name = self.config['model_id']

        dashboard = {
            "widgets": []
        }

        hardware = {"type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 6,
                    "height": 6,
                    "properties": {"view": "timeSeries",
                                   "stacked": False,
                                   "metrics": [["/aws/sagemaker/Endpoints",
                                                "CPUUtilization",
                                                "EndpointName",
                                                self.config['model_id'],
                                                "VariantName",
                                                "AllTraffic"],
                                               [".",
                                                "MemoryUtilization",
                                                ".",
                                                ".",
                                                ".",
                                                "."],
                                               [".",
                                                "DiskUtilization",
                                                ".",
                                                ".",
                                                ".",
                                                "."]],
                                   "region": self.config["region"]}}

        invocations = {"type": "metric",
                       "x": 6,
                       "y": 0,
                       "width": 6,
                       "height": 6,
                       "properties": {"view": "timeSeries",
                                      "stacked": False,
                                      "metrics": [["AWS/SageMaker",
                                                   "Invocation5XXErrors",
                                                   "EndpointName",
                                                   self.config['model_id'],
                                                   "VariantName",
                                                   "AllTraffic"],
                                                  [".",
                                                   "Invocation4XXErrors",
                                                   ".",
                                                   ".",
                                                   ".",
                                                   "."],
                                                  [".",
                                                   "Invocations",
                                                   ".",
                                                   ".",
                                                   ".",
                                                   "."],
                                                  [".",
                                                   "InvocationsPerInstance",
                                                   ".",
                                                   ".",
                                                   ".",
                                                   "."]],
                                      "region": self.config["region"]}}

        latency = {"type": "metric",
                   "x": 12,
                   "y": 0,
                   "width": 6,
                   "height": 6,
                   "properties": {"view": "timeSeries",
                                  "stacked": False,
                                  "metrics": [["AWS/SageMaker",
                                               "ModelLatency",
                                               "EndpointName",
                                               self.config['model_id'],
                                               "VariantName",
                                               "AllTraffic"],
                                              [".",
                                               "OverheadLatency",
                                               ".",
                                               ".",
                                               ".",
                                               "."]],
                                  "region": self.config["region"]}}

        logs = {
            "type": "log",
            "x": 0,
            "y": 6,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/sagemaker/Endpoints/{}' | fields @timestamp, @message\n| sort @timestamp desc\n| limit 20".format(
                    self.config['model_id']),
                "region": self.config["region"],
                "stacked": False,
                "view": "table"}}

        dashboard['widgets'].append(hardware)
        dashboard['widgets'].append(invocations)
        dashboard['widgets'].append(latency)
        dashboard['widgets'].append(logs)

        response = cw.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard)
        )

        logging.info(f'response: {response}')
        output = {
            "dashboard": {
                "cloudwatch_dashboard_name": dashboard_name,
                "end_date": str(datetime.now())
            }
        }

        self.config["dynamodb_item"]["steps"]["monitoring"] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def monitor_data_quality(self):
        logging.info("#############STARTING DATA QUALITY#############")

        s3_output_path = f"s3://{self.config['bucket_models']}/{self.config['monitoring']['output_prefix']}/{self.config['model_id']}/data-quality"
        baseline_dataset = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_dataset_file"]

        instance_count = self.config['monitoring']['data_quality'][
            'instance_count'] if 'instance_count' in self.config['monitoring']['data_quality'] else 1
        instance_type = self.config['monitoring']['data_quality'][
            'instance_type'] if 'instance_type' in self.config['monitoring']['data_quality'] else "ml.m5.xlarge"
        volume_size_in_gb = self.config['monitoring']['data_quality'][
            'volume_size_in_gb'] if 'volume_size_in_gb' in self.config['monitoring']['data_quality'] else 20
        max_runtime_in_seconds = self.config['monitoring']['data_quality'][
            'max_runtime_in_seconds'] if 'max_runtime_in_seconds' in self.config['monitoring']['data_quality'] else 3600
        tags = self.get_model_tags()

        monitor_job_name = "MonDq-{}-{}".format(
            datetime.now().strftime(self.date_format),
            self.config['model_id'])
        fixed_monitor_job_name = monitor_job_name[:63]

        my_default_monitor = DefaultModelMonitor(
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=fixed_monitor_job_name,
            output_kms_key=self.config['s3_kms_id'],
            tags=tags
        )

        logging.info(f'baseline_dataset: {baseline_dataset}')
        logging.info(f's3_output_path: {s3_output_path}')

        my_default_monitor.suggest_baseline(
            job_name=fixed_monitor_job_name,
            baseline_dataset=baseline_dataset,
            dataset_format=self.baseline_dataset_format,
            output_s3_uri=s3_output_path,
            wait=True,
            logs=True
        )

        logging.info(
            "#############STARTING DATA QUALITY SCHEDULE#############")

        from sagemaker.model_monitor import CronExpressionGenerator
        from time import gmtime, strftime

        endpoint_input = self.config["dynamodb_item"]["steps"]["deploy"]["endpoint_name"]
        s3_report_path = s3_output_path + '/schedule-results'

        schedule_job_name = "MonSchDq-{}-{}".format(
            datetime.now().strftime(self.date_format),
            self.config['model_id'])
        fixed_schedule_job_name = schedule_job_name[:63]

        my_default_monitor.create_monitoring_schedule(
            monitor_schedule_name=fixed_schedule_job_name,
            endpoint_input=endpoint_input,
            record_preprocessor_script=self.data_quality_s3_preprocessor_uri,
            post_analytics_processor_script=self.data_quality_s3_postprocessor_uri,
            output_s3_uri=s3_report_path,
            statistics=my_default_monitor.baseline_statistics(),
            constraints=my_default_monitor.suggested_constraints(),
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            enable_cloudwatch_metrics=True)

        output = {
            "data_quality": {
                "model_monitor_job_name": fixed_monitor_job_name,
                "monitoring_schedule_job_name": fixed_schedule_job_name,
                "s3_output_path": s3_output_path,
                "end_date": str(datetime.now())
            }
        }

        self.config["dynamodb_item"]["steps"]["monitoring"] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])

    def monitor_model_quality(self):
        logging.info("#############STARTING MODEL QUALITY#############")
        s3_output_path = f"s3://{self.config['bucket_models']}/{self.config['monitoring']['output_prefix']}/{self.config['model_id']}/model-quality"
        baseline_dataset = self.config["dynamodb_item"]["steps"]["feature_engineering"]["s3_dataset_file"]

        instance_count = self.config['monitoring']['model_quality'][
            'instance_count'] if 'instance_count' in self.config['monitoring']['model_quality'] else 1
        instance_type = self.config['monitoring']['model_quality'][
            'instance_type'] if 'instance_type' in self.config['monitoring']['model_quality'] else "ml.m5.xlarge"
        volume_size_in_gb = self.config['monitoring']['model_quality'][
            'volume_size_in_gb'] if 'volume_size_in_gb' in self.config['monitoring']['model_quality'] else 20
        max_runtime_in_seconds = self.config['monitoring']['model_quality'][
            'max_runtime_in_seconds'] if 'max_runtime_in_seconds' in self.config['monitoring']['model_quality'] else 3600
        tags = self.get_model_tags()

        monitor_job_name = "MonMq-{}-{}".format(
            datetime.now().strftime(self.date_format),
            self.config['model_id'])
        fixed_monitor_job_name = monitor_job_name[:63]

        model_quality_monitor = ModelQualityMonitor(
            role=self.config['role'],
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            max_runtime_in_seconds=max_runtime_in_seconds,
            output_kms_key=self.config['s3_kms_id'],
            base_job_name=fixed_monitor_job_name,
            tags=tags
        )

        problem_type = self.config['monitoring']['model_quality'][
            'problem_type'] if 'problem_type' in self.config['monitoring']['model_quality'] else None
        inference_attribute = self.config['monitoring']['model_quality'][
            'inference_attribute'] if 'inference_attribute' in self.config['monitoring']['model_quality'] else None
        probability_attribute = self.config['monitoring']['model_quality'][
            'probability_attribute'] if 'probability_attribute' in self.config['monitoring']['model_quality'] else None
        ground_truth_attribute = self.config['monitoring']['model_quality'][
            'ground_truth_attribute'] if 'ground_truth_attribute' in self.config['monitoring']['model_quality'] else None

        model_quality_monitor.suggest_baseline(
            job_name=fixed_monitor_job_name,
            baseline_dataset=baseline_dataset,
            dataset_format=self.baseline_dataset_format,
            output_s3_uri=s3_output_path,
            problem_type=problem_type,
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            ground_truth_attribute=ground_truth_attribute,
            wait=True,
            logs=False
        )

        logging.info(
            "#############STARTING MODEL QUALITY SCHEDULE#############")

        from sagemaker.model_monitor import CronExpressionGenerator
        from time import gmtime, strftime

        endpoint_input = self.config["dynamodb_item"]["steps"]["deploy"]["endpoint_name"]
        s3_report_path = s3_output_path + '/schedule-results'

        schedule_job_name = "MonSchMq-{}-{}".format(
            datetime.now().strftime(self.date_format),
            self.config['model_id'])
        fixed_schedule_job_name = schedule_job_name[:63]

        model_quality_monitor.create_monitoring_schedule(
            monitor_schedule_name=fixed_schedule_job_name,
            endpoint_input=endpoint_input,
            problem_type=problem_type,
            record_preprocessor_script=self.model_quality_s3_preprocessor_uri,
            post_analytics_processor_script=self.model_quality_s3_postprocessor_uri,
            output_s3_uri=s3_report_path,
            # statistics=model_quality_monitor.baseline_statistics(),
            constraints=model_quality_monitor.suggested_constraints(),
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            ground_truth_input=baseline_dataset,
            enable_cloudwatch_metrics=True)

        output = {
            "model_quality": {
                "model_monitor_job_name": fixed_monitor_job_name,
                "monitoring_schedule_job_name": fixed_schedule_job_name,
                "s3_output_path": s3_output_path,
                "end_date": str(datetime.now())
            }
        }

        self.config["dynamodb_item"]["steps"]["monitoring"] = output
        utils.update_dynamodb_item(
            self.config["dynamodb_table"],
            self.config["dynamodb_item"])
