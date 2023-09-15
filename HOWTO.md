# HOW-TO

## Compile and Push a new version to ECR

### Requirements
1. aws cli
2. docker

### Procedure

1. Login into ECR:

    `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 204991841662.dkr.ecr.us-east-1.amazonaws.com`

2. Build the docker image standing from the root directory of this repository (the version `latest` can be change to another if needed. Consider that the fargate revision task should match with the version):

    `docker build -t dev/services/balanz-mlops:latest .`

3. Tag the image

    `docker tag dev/services/balanz-mlops:latest 204991841662.dkr.ecr.us-east-1.amazonaws.com/dev/services/balanz-mlops:latest`

4. Push into ECR

    `docker push 204991841662.dkr.ecr.us-east-1.amazonaws.com/dev/services/balanz-mlops:latest`

5. Optional: use the script [buildpush.sh](buildpush.sh) instead to execute this procedure.


## Execute an individual step without pipeline

1. Change the task definition name variable in [run_task.sh](run_task.sh) script. 
    
    Example: 
    
    `task_definition="dev-als-recommendations-securityid"`

2. Update the corresponding revision in the revision variable. 
    
    Example: 
    
    `revision=6`

3. Set up the model_id that is going to be executed. 
    
    Example: 
    
    `model_id="testing"`

4. Run the [run_task.sh](run_task.sh) script specifying the step you want to execute. Example:

	`./run_task.sh train`

	`./run_task.sh delete`

	`./run_task.sh feature_engineering`

	`./run_task.sh post_processing`

5. This will execute a new fargate task outside step functions.

6. (Optional)In case we just want to execute a specific step on a specific model id, run the [run_step.sh](run_step.sh) script specifying the step ,task_definition, revision and model_id you want to execute. Example:

	`./run_step.sh deploy dev-operations-recommendations-securityid 1 operations-recommendations-securityid-d5130808-46f6-11ec-a354-ab8632b095b5`





