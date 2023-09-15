import os
import logging
from models.model import Model
import libs.utils as utils
import sys
import json
import importlib
import traceback

utils.set_logger(
    os.getenv("log_level") if os.getenv("log_level") is not None
    else "INFO")


def model_factory(class_name, model_config) -> Model:
    # models should be created under this path (src/models/)
    module_name = "models." + class_name
    module = importlib.import_module(module_name)
    # the name of the class should match the name of the file
    class_ = getattr(module, class_name)
    instance = class_(class_name, model_config)
    return instance


class StepSelector:

    def switch(self, model, step):
        return getattr(self, step, self.default)(model)

    def default(self, model):
        raise Exception("Invalid step")

    def feature_engineering(self, model):
        model.feature_engineering()

    def train(self, model):
        model.tuner() if config['enable_tuner'] else model.train()

    def registry(self, model):
        model.registry() if config['enable_registry'] else logging.info(
            "Registry disabled")

    def validate(self, model):
        model.validate() if config['enable_validate'] else logging.info(
            "Validation disabled")

    def deploy(self, model):
        model.deploy() if config['enable_deploy'] else logging.info(
            "Deployment disabled")

    def monitoring(self, model):
        model.monitoring() if config['enable_monitoring'] else logging.info(
            "Monitoring disabled")

    def post_processing(self, model):
        model.post_processing() if config['enable_post_processing'] else logging.info(
            "Post Processing disabled")

    def delete(self, model):
        model.delete()


def process_model_by_name(config, model_name, step):
    dynamodb_item = utils.get_dynamodb_item(config)
    config["dynamodb_item"] = dynamodb_item
    model = model_factory(model_name.replace("-", "_"), config)
    if model is not None:
        step_selector = StepSelector()
        step_selector.switch(model, step)
    else:
        raise Exception("Unknown model name")


try:
    model_name = os.environ.get('BALANZ_MODEL_NAME')

    config = utils.get_dict_from_s3_json(
        os.environ.get('BALANZ_CONFIG_BUCKET'),
        f'configs/{model_name}.json')

    config['sns_topic_arn'] = os.environ.get('BALANZ_SNS_TOPIC_ARN')
    config['s3_kms_id'] = os.environ.get('BALANZ_S3_KMS_ID')
    config['region'] = os.environ.get('AWS_REGION')
    config['config_bucket'] = os.environ.get('BALANZ_CONFIG_BUCKET')
    config['database'] = os.environ.get('BALANZ_GLUE_DATABASE_NAME')
    config['bucket_source'] = os.environ.get(
        'BALANZ_ATHENA_RESULTS_S3_BUCKET_NAME')
    config['bucket_source_prefix'] = os.environ.get(
        'BALANZ_ATHENA_RESULTS_S3_BUCKET_PREFIX_PATH')
    config['bucket_models'] = os.environ.get('BALANZ_MODELS_BUCKET_NAME')
    config['workgroup'] = os.environ.get('BALANZ_ATHENA_WORKGROUP_NAME')
    config['role'] = os.environ.get('BALANZ_FARGATE_TASK_ROLE')
    config['step'] = os.environ.get('STEP')
    config['model_id'] = os.environ.get('model_id')
    config['model_name'] = model_name

    if config['step'] is None:
        raise Exception("Step not specified")

    if config['model_id'] is None:
        raise Exception("ModelId not specified")

    logging.info(f'model_name: {model_name}')
    logging.info(f"model_id: {config['model_id']}")
    logging.info(f'config: {config}')

    process_model_by_name(config, model_name, config['step'])
    sys.exit(0)

except Exception as e:
    logging.error(f'Step failed: {traceback.format_exc()}')
    message = {
        "Model": os.environ.get('BALANZ_MODEL_NAME'),
        "Model ID": os.environ.get('model_id'),
        "Step": os.environ.get('STEP'),
        "Status": "FAILED",
        "Error": str(e)
    }

    utils.send_sns_notification(
        os.environ.get('BALANZ_SNS_TOPIC_ARN'), message)
    sys.exit(2)
