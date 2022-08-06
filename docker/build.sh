#!/bin/bash

repo=gitlab-master.nvidia.com:5005/tkurth/era5_wind
#tag=latest
#tag=debug
tag=jaideep_legacy_dataloader

cd ../

# build
docker build -t ${repo}:${tag} -f docker/Dockerfile .

# push
docker push ${repo}:${tag}

# retag and repush
#docker tag ${repo}:${tag} thorstenkurth/era5-wind:${tag}
#docker push thorstenkurth/era5-wind:${tag}
