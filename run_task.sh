#!/bin/bash

step=$1
task_definition="dev-operations-recommendations-securityid"

revision=1

model_id="operations-recommendations-securityid-d5130808-46f6-11ec-a354-ab8632b095b5"

aws ecs run-task \
	--cluster  dev-balanz-mlops \
	--task-definition $task_definition:$revision \
	--overrides '{"containerOverrides": [{"name":"'$task_definition'", "environment": [{"name": "STEP", "value": "'$step'"}, {"name": "model_id", "value": "'$model_id'"}]}]}' \
	--network-configuration '{"awsvpcConfiguration": {"subnets": ["subnet-046eb519c1ab2822a","subnet-035da4a4ca17b72b4","subnet-05bae5faa2d5bf72f"], "securityGroups": ["sg-00499125cafc57903"]}}' \
	--region 'us-east-1'