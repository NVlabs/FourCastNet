#!/bin/bash

#fail if any command fails
set -e pipefail

# make required directories if they don't exist

if [ ! -d ${PWD}/gribfiles ]; then
    mkdir -p ${PWD}/gribfiles
fi

if [ ! -d ${PWD}/inventory ]; then
    mkdir -p ${PWD}/inventory
fi

if [ ! -d ${PWD}/weights ]; then
    mkdir -p ${PWD}/weights
fi

if [ ! -d ${PWD}/perlscripts ]; then
    mkdir -p ${PWD}/perlscripts
fi

if [ ! -f ./weights/backbone_v0.1.ckpt ]; then
    echo "backbone_v0.1.ckpt not found in checkpoints. Downloading..."
    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/backbone_v0.1.ckpt -P ./weights/
fi

if [ ! -f ./weights/global_means.npy ]; then
    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/stats_v0.1/global_means.npy -P ./weights/
    wget https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/stats_v0.1/global_stds.npy -P ./weights/
fi

# check that the NOAA perl scripts get_inv.pl and get_grib.pl are present
if [ ! -f ./perlscripts/get_inv.pl ]; then
    echo "perlscripts not found. please run the following and chmod u+x the scripts:
    wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/fast_downloading_grib/get_inv.pl -P ./perlscripts/
    wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/fast_downloading_grib/get_grib.pl -P ./perlscripts/"
    exit 1
fi

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
