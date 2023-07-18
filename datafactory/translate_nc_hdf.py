# copy downloaded data from nc to hdf

import os
import argparse

import h5py
from netCDF4 import Dataset as DS


def writetofile(src, dest, channel_idx, variable_name, src_idx=0, frmt='nc'):
    if not os.path.isfile(src):
        return "did not find source file"

    fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
    with h5py.File(dest, 'a') as f:
        if 'fields' not in f:
            f.create_dataset('fields',
                             shape=(fsrc.shape[0], 20, fsrc.shape[1],
                                    fsrc.shape[2]),
                             dtype='<f4')
    fdest = h5py.File(dest, 'a')

    BATCH = 2**4
    END = fsrc.shape[0]
    idx = 0
    while idx < END:
        if END - idx < BATCH:
            if len(fsrc.shape) == 4:
                ims = fsrc[idx:END, src_idx]
            else:
                ims = fsrc[idx:END]
            fdest['fields'][idx:END, channel_idx, :, :] = ims
            break
        else:
            if len(fsrc.shape) == 4:
                ims = fsrc[idx:idx + BATCH, src_idx]
            else:
                ims = fsrc[idx:idx + BATCH]
            fdest['fields'][idx:idx + BATCH, channel_idx, :, :] = ims
            idx += BATCH


def writeall(filename, path):
    dest = f'{path}/{filename}.h5'

    src = f'{path}/{filename}_sfc.nc'
    writetofile(src, dest, 0, 'u10')  # u10
    writetofile(src, dest, 1, 'v10')  # v10
    writetofile(src, dest, 2, 't2m')  # t2m
    writetofile(src, dest, 3, 'sp')  # sp
    writetofile(src, dest, 4, 'msl')  # mslp

    src = f'{path}/{filename}_pl.nc'
    writetofile(src, dest, 5, 't', 2)  # t850

    writetofile(src, dest, 6, 'u', 3)  # u1000
    writetofile(src, dest, 7, 'v', 3)  # v1000
    writetofile(src, dest, 8, 'z', 3)  # z1000

    writetofile(src, dest, 9, 'u', 2)  # u850
    writetofile(src, dest, 10, 'v', 2)  # v850
    writetofile(src, dest, 11, 'z', 2)  # z850

    writetofile(src, dest, 12, 'u', 1)  # u500
    writetofile(src, dest, 13, 'v', 1)  # v500
    writetofile(src, dest, 14, 'z', 1)  # z500

    writetofile(src, dest, 15, 't', 1)  # t500
    writetofile(src, dest, 16, 'z', 0)  # z50

    writetofile(src, dest, 17, 'r', 1)  # r500
    writetofile(src, dest, 18, 'r', 2)  # r850

    src = f'{path}/{filename}_sfc.nc'
    writetofile(src, dest, 19, 'tcwv')  # tcwv
    # writetofile(src, dest, 20, 'sst') # sst


if __name__ == '__main__':
    # -- prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--datapath", type=str)

    # -- parse config file and script args
    args = parser.parse_args()

    # -- run
    writeall(args.filename, args.datapath)
