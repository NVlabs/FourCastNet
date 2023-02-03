#!/bin/bash

#fail if any command fails
set -e pipefail

#check if backbone_v0.1.ckpt exists in checkpoints
#if [ ! -f ./weights/backbone_v0.1.ckpt ]; then
#    echo "backbone_v0.1.ckpt not found in checkpoints. Downloading..."
#    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/backbone_v0.1.ckpt -P ./weights/
#fi
#
#if [ ! -f ./weights/global_means.npy ]; then
#    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/stats_v0.1/global_means.npy -P ./weights/
#    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/stats_v0.1/global_stds.npy -P ./weights/
#fi
#
##get grib
YEAR=`date +%Y`
MONTH=`date +%m`
DAY=`date +%d`
HOUR=`date +%H`
FORECAST_HOUR=$(( 10#6 * (10#$HOUR / 10#6) ))
DATESTR=$YEAR-$MONTH-$DAY-$FORECAST_HOUR

./getGFS.sh $YEAR $MONTH $DAY $HOUR $FORECAST_HOUR
##
#convert
source activate grib
python grb2hdf5.py
conda deactivate
#
#forecast
python real_time_inference.py --yaml_config ../config/AFNO.yaml --config afno_backbone_26ch 
#

#plot
source activate grib
python plot.py $DATESTR 
conda deactivate

#push to dashboard
#git add images
#git commit -m "update"
#git push
