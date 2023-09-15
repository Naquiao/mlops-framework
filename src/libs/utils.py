import os
import boto3
import glob
from datetime import datetime
import types
import re
import time as ttime
import codecs
import csv
import json
import jsonlines
from dateutil.relativedelta import relativedelta
import math
import logging
import botocore
from boto3.dynamodb.conditions import Key, Attr
import pandas as pd
from io import StringIO
from io import BytesIO
import awswrangler as wr
import sagemaker.amazon.common as smac
import sys


def set_logger(log_level):
    if log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)
    elif log_level == "ERROR":
        logging.basicConfig(level=logging.ERROR)
    elif log_level == "CRITICAL":
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO)


def read_csv_from_s3(key, env_variables):
    bucket_name = env_variables['bucket_source']
    prefix = env_variables['bucket_source_prefix']
    client = boto3.client("s3")
    filename = '/tmp/output.csv'
    fkey = prefix + '/' + key
    print(f'bucket name: {bucket_name}   key:{fkey}')
    client.download_file(bucket_name, fkey, filename)
    rows = []
    data = csv.DictReader(open(filename))
    for row in data:
        rows.append(row)
    return rows


def get_dict_from_s3_json(bucket, key):
    client = boto3.client("s3")
    result = client.get_object(Bucket=bucket, Key=key)
    j = json.loads(result['Body'].read())
    return j


def read_file_from_bucket(bucketname, filename):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucketname, filename)
    body = obj.get()['Body'].read().decode('utf-8')
    return body


def make_athena_query(query, env_variables):

    params = {
        'region': env_variables['region'],
        'database': env_variables['database'],
        'bucket': env_variables['bucket_source'],
        'query': query,
        'workgroup': env_variables['workgroup']
    }

    s3_filename = athena_to_s3(params)

    return s3_filename


def athena_to_s3(params):

    session = boto3.Session()
    client = session.client('athena', region_name=params["region"])
    execution = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']},
        # ATHENA DATABASE NAME
        WorkGroup=params['workgroup'])  # WORKGROUP, THIS DEFINES THE OUTPUT BUCKET WHERE THE QUERY RESULTS ARE GOING TO BE STORED

    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    # THIS IS IMPLEMENTED AS A BUSY LOOP, COULD BE ALSO IMPLEMENTED AS AN ASYNC FUNCTION
    # BUT MAYBE ITS AND OVERKILL SINCE THERE IS ONLY ONE LINE OF EXECUTION

    while (state in ['RUNNING', 'QUEUED']):
        response = client.get_query_execution(QueryExecutionId=execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state in ('FAILED', 'CANCELLED'):
                logging.error(response)
                raise Exception("Query execution failed: {}".format(
                    response['QueryExecution']["Status"]["StateChangeReason"]))
            elif state == 'SUCCEEDED':
                logging.debug(response)
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*/(.*)', s3_path)[0]
                logging.info(f'FILENAME: {filename}')
                return filename
        ttime.sleep(10)


def put_obj_in_bucket(obj, encoding, bucket, key, file_format, header=False):
    logging.info(f'put_obj_in_bucket')
    s3 = boto3.client('s3')

    # JSON #
    if "json" == file_format:
        upload = s3.put_object(
            Body=json.dumps(obj).encode(encoding),
            Bucket=bucket,
            Key=key
        )
    # JSONLINES #
    if "jsonl" == file_format:
        with jsonlines.open('temp.jsonl', 'w') as writer:
            writer.write_all(obj)
        upload = s3.upload_file(
            Filename='temp.jsonl',
            Bucket=bucket,
            Key=key
        )
    # CSV #
    if "csv" == file_format:
        csv_buffer = StringIO()
        obj.to_csv(csv_buffer, index=False, header=header)
        upload = s3.put_object(
            Body=csv_buffer.getvalue(),
            Bucket=bucket,
            Key=key
        )
    # PARQUET #
    if "parquet" == file_format:
        upload = f's3://{bucket}/{key}'
        wr.s3.to_parquet(
            df=obj,
            path=upload,
            compression='snappy'
        )
    # PROTOBUF SPARSE #
    if "protobuf-sparse" == file_format:
        buf = BytesIO()
        smac.write_spmatrix_to_sparse_tensor(buf, obj[0], obj[1])
        buf.seek(0)
        boto3.resource('s3').Bucket(bucket).Object(key).upload_fileobj(buf)
        upload = 's3://{}/{}'.format(bucket, key)
    # PROTOBUF DENSE #
    if "protobuf-dense" == file_format:
        buf = BytesIO()
        smac.write_numpy_to_dense_tensor(buf, obj[0], obj[1])
        buf.seek(0)
        boto3.resource('s3').Bucket(bucket).Object(key).upload_fileobj(buf)
        upload = 's3://{}/{}'.format(bucket, key)

    return upload


def send_sns_notification(sns_topic_arn, message):
    sns = boto3.client('sns')
    logging.info(f'message to post to sns: {message}')
    try:
        response = sns.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps({'default': json.dumps(message)}),
            MessageStructure='json'
        )
        logging.info(f'sens_sns_notification response: {response}')
    except botocore.exceptions.ClientError as e:
        logging.error("Error sending SNS topic: {}".format(e))


def get_dynamodb_item(config):
    # If item exists in dynamodb we fetch it, If not we create a new one with
    # hash key model_id
    table_name = config["dynamodb_table"]
    dynamodb_key = "model_id"
    hashkey = config["model_id"]

    dynamodb = boto3.resource('dynamodb', os.environ['AWS_REGION'])

    response = dynamodb.batch_get_item(
        RequestItems={
            table_name: {
                'Keys': [{dynamodb_key: hashkey}],
            }
        },
        ReturnConsumedCapacity='NONE'
    )
    logging.info(f'dynamodb response: {response}')
    existing_item = response["Responses"][table_name]

    if len(existing_item) == 0:
        item = create_registry_in_dynamo(table_name, hashkey, datetime.now())
    else:
        item = existing_item[0]
    return item


def create_registry_in_dynamo(table_name, hashkey, date):
    dynamodb = boto3.resource('dynamodb', os.environ['AWS_REGION'])
    table = dynamodb.Table(table_name)

    item = {
        'model_id': hashkey,
        # this could be change to epoch instead of date
        'creation_date': str(date),
        'steps': {}
    }

    data = table.put_item(
        Item=item
    )

    logging.info(f'data after put item: {data}')

    return item


def update_dynamodb_item(table_name, item):
    dynamodb = boto3.resource('dynamodb', os.environ['AWS_REGION'])
    table = dynamodb.Table(table_name)
    logging.info(f'item: {item}')
    upload = table.put_item(
        Item=item
    )
    logging.info(f'UPLOAD INFO: {upload}')


def find_production_model_in_dynamodb(table_name):
    dynamodb_key = "is_production"

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    response = table.query(
        IndexName='is_production-gsi-index',
        KeyConditionExpression=Key(dynamodb_key).eq("true")
    )

    logging.info(f'dynamodb response: {response}')
    existing_item = response["Items"]

    if len(existing_item) == 0:
        return None
    else:
        item = existing_item[0]
    return item


def sagemaker_delete_monitoring_schedules(endpoint_name):
    try:
        logging.info(f'deleting monitoring schedules: {endpoint_name}')
        client = boto3.client('sagemaker')
        retry = 3
        empty = False
        while empty == False and retry > 0:
            retry -= 1
            response = client.list_monitoring_schedules(
                EndpointName=endpoint_name)
            if len(response['MonitoringScheduleSummaries']) == 0:
                empty = True
            else:
                for monsch in response['MonitoringScheduleSummaries']:
                    response = client.delete_monitoring_schedule(
                        MonitoringScheduleName=monsch['MonitoringScheduleName'])
                    logging.info(
                        f'delete {monsch["MonitoringScheduleName"]} response: {response}')
                ttime.sleep(10)
    except Exception as e:
        logging.info(f'delete monitoring schedules error: {e}')


def sagemaker_delete_model(model_name):
    try:
        logging.info(f'deleting model: {model_name}')
        client = boto3.client('sagemaker')
        response = client.delete_model(ModelName=model_name)
        logging.info(f'delete model response: {response}')
    except Exception as e:
        logging.info(f'delete model error: {e}')


def sagemaker_delete_endpoint(endpoint_name):
    try:
        logging.info(f'deleting endpoint: {endpoint_name}')
        client = boto3.client('sagemaker')
        response = client.delete_endpoint(EndpointName=endpoint_name)
        logging.info(f'delete endpoint response: {response}')
    except Exception as e:
        logging.info(f'delete endpoint error: {e}')


def sagemaker_delete_endpoint_config(endpoint_config_name):
    try:
        logging.info(f'deleting endpoint config: {endpoint_config_name}')
        client = boto3.client('sagemaker')
        response = client.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name)
        logging.info(f'delete endpoint config response: {response}')
    except Exception as e:
        logging.info(f'delete endpoint config error: {e}')


def cloudwatch_delete_dashboard(endpoint_name):
    try:
        logging.info(f'deleting cloudwatch dashboard: {endpoint_name}')
        client = boto3.client('cloudwatch')
        response = client.delete_dashboards(
            DashboardNames=[endpoint_name])
        logging.info(f'delete cloudwatch dashboard response: {response}')
    except Exception as e:
        logging.info(f'delete cloudwatch dashboard error: {e}')


def get_fargate_task_definition_arn(taskDefinition):
    try:
        client = boto3.client('ecs')
        response = client.describe_task_definition(
            taskDefinition=taskDefinition
        )
        return response['taskDefinition']['taskDefinitionArn']
    except Exception as e:
        logging.info(f'get fargate task definition arn error: {e}')


def start_stepfunction(stateMachineArn, name, input_json):
    try:
        stepfunctions = boto3.client('stepfunctions')
        response = stepfunctions.start_execution(
            stateMachineArn=stateMachineArn,
            name=name,
            input=json.dumps(input_json)
        )
        return response
    except Exception as e:
        logging.info(f'start stepfunction error: {e}')
