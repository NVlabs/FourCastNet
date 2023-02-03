#!/bin/bash

#obtain perlscripts for downloading grib files from NCEP
if [ ! -d ${PWD}/perlscripts ]; then
    echo "perlscripts not found. Downloading..."
    wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/fast_downloading_grib/get_inv.pl -P ./perlscripts/
    wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/fast_downloading_grib/get_grib.pl -P ./perlscripts/
fi

#get YEAR, MONTH, DAY, and FORECAST_HOUR from command line
YEAR=$1
MONTH=$2
DAY=$3
HOUR=$4
FORECAST_HOUR=$5

DATESTR=$YEAR$MONTH$DAY
printf -v HH "%02d" $FORECAST_HOUR

URL='https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.'$DATESTR'/'$HH'/atmos/gfs.t'$HH'z.pgrb2.0p25.f000'
INVENTORY_PATH=${PWD}/inventory
PERL_PATH=${PWD}/perlscripts
GRIB_PATH=${PWD}/gribfiles
mkdir -p $GRIB_PATH
$PERL_PATH/get_inv.pl $URL.idx > $INVENTORY_PATH/inv
INV=$INVENTORY_PATH/inv

grep ":UGRD:10 m above ground" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/u10.grb
grep ":VGRD:10 m above ground" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/v10.grb
grep ":UGRD:100 m above ground" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/u100.grb
grep ":VGRD:100 m above ground" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/v100.grb
grep ":TMP:2 m above ground" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/t2m.grb
grep ":PRES:surface" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/sp.grb
grep ":PRMSL:" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/mslp.grb
grep -E ":(HGT|TMP|UGRD|VGRD|RH):(1000 mb|850 mb|500 mb|250 mb|50 mb):" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/tuvzr.grb
grep ":PWAT:entire atmosphere" < $INV | $PERL_PATH/get_grib.pl $URL $GRIB_PATH/tcwv.grb
