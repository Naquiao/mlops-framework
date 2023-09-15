# Fargate Environment Variables


| Key | Description | Example |
| --- |------------ | ------- |
|**AWS_REGION** | AWS Region where the resources will start and run |	us-east-1 |
|**BALANZ_ATHENA_RESULTS_S3_BUCKET_NAME** | S3 bucket name where query output will be stored |	balanz-mlops-dev-models |
|**BALANZ_ATHENA_RESULTS_S3_BUCKET_PREFIX_PATH** | S3 prefix where query outputs will be stored |	output |
|**BALANZ_ATHENA_WORKGROUP_NAME** | Athena workgroup name |	dev-models |
|**BALANZ_CLOUDWATCH_LOG_GROUP_PREFIX** | Cloudwatch log group prefix. This will be after completed with the model name. |	models/dev/balanz-mlops |
|**BALANZ_CONFIG_BUCKET** | S3 bucket name where Model Config file is stored |	balanz-mlops-dev-models |
|**BALANZ_DYNAMO_TABLE** | DynamoDB table name to save model information and steps outputs |	dev-models-user-recommendations |
|**BALANZ_ENVIRONMENT** | CURRENTLY NOT IN USE |	dev |
|**BALANZ_FARGATE_TASK_ROLE** | Fargate task IAM Role |	arn:aws:iam::204991841662:role/dev-user-recommendations-role |
|**BALANZ_GLUE_DATABASE_NAME** | Glue database name where feature engineering's query will be made |    dev-datalake-balanz-rds-raw-db |
|**BALANZ_MODEL_NAME** | Name of the model to execute. This should match with the class name and class file |	user-recommendations |
|**BALANZ_MODELS_BUCKET_NAME** | S3 bucket name where all the models file will be stored |	balanz-mlops-dev-models |
|**BALANZ_S3_KMS_ID** | KMS Key ID used to encrypt/decrypt objects from models bucket |	371c7544-33da-435b-aca8-66725939e65a |
|**BALANZ_SNS_TOPIC_ARN** | SNS Topic ARN where notifications will be pushed |	arn:aws:sns:us-east-1:204991841662:dev-balanz-mlops-notify-teams-sns |
