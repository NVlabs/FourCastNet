#REGISTRY ?= us.gcr.io/vcm-ml
VERSION ?= $(shell git rev-parse HEAD)
IMAGE ?= fourcastnet

build_docker_image:
	docker build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

build_beaker_image: build_docker_image
	beaker image create --name $(IMAGE)-$(VERSION) $(IMAGE):$(VERSION)

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash

train: build_docker_image
	docker run \
		--rm \
		-w /opt/ERA5_wind \
		--mount type=bind,source=$(shell pwd)/config,target=/opt/ERA5_wind/config \
		--mount type=bind,source=/Users/oliverwm/Documents/FourCastNet-data,target=/data \
		$(IMAGE):$(VERSION) \
		python train.py --yaml_config config/AFNO.yaml --config full_field --enable_amp
