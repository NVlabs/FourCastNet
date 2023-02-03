#!/bin/bash

repo=gitlab-master.nvidia.com:5005/jpathak/fourcastnet
tag=latest

cd ../

# build
docker build --platform=linux/amd64 -t ${repo}:${tag} -f docker/Dockerfile .

# push
docker push ${repo}:${tag}
