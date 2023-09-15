#!/bin/sh


aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 204991841662.dkr.ecr.us-east-1.amazonaws.com
docker build -t dev/services/balanz-mlops:latest .
docker tag dev/services/balanz-mlops:latest 204991841662.dkr.ecr.us-east-1.amazonaws.com/dev/services/balanz-mlops:latest
read -n1 -rsp $'Press any key to push or Ctrl+C to exit...\n'
docker push 204991841662.dkr.ecr.us-east-1.amazonaws.com/dev/services/balanz-mlops:latest
