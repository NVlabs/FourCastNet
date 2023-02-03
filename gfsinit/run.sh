#!/bin/bash

#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

#add parent directory to python path

export PYTHONPATH="${PYTHONPATH}:.."

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
