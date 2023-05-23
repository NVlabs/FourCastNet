# copy downloaded data from nc to hdf

# instructions: 
# set NIMGTOT correctly

import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

DATE = '{{= filename }}'
PATH = '{{= datapath }}'

def writetofile(src, dest, channel_idx, variable_name, src_idx=0, frmt='nc'):
    if not os.path.isfile(src):
        return "did not find source file"

    BATCH = 2**4
    RANK = MPI.COMM_WORLD.RANK
    NPROC = MPI.COMM_WORLD.size
    NIMGTOT = 4 # src_shape[0]
    PREDICTION_LENGTH = 1

    NIMG = NIMGTOT//NPROC
    BASE = RANK*NIMG
    END = (RANK+1)*NIMG if RANK<NPROC - 1 else NIMGTOT
    idx = BASE

    fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
    with h5py.File(dest, 'w') as f:
        f.create_dataset('fields', shape=(PREDICTION_LENGTH, 21, fsrc.shape[1], fsrc.shape[2]), dtype='<f4')
    fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

    # try: 
    #     fdest['fields']
    # except:
    #     new_dataset = fdest.create_dataset('fields', (NIMGTOT, 21, fsrc.shape[1], fsrc.shape[2]), dtype='<f4')
    #     del new_dataset

    start = time.time()
    while idx<END:
        if END - idx < BATCH:
            if len(fsrc.shape) == 4:
                ims = fsrc[idx:END,src_idx]
            else:
                ims = fsrc[idx:END]
            fdest['fields'][idx:END, channel_idx, :, :] = ims
            break
        else:
            if len(fsrc.shape) == 4:
                ims = fsrc[idx:idx+BATCH,src_idx]
            else:
                ims = fsrc[idx:idx+BATCH]
            fdest['fields'][idx:idx+BATCH, channel_idx, :, :] = ims
            idx+=BATCH
            ttot = time.time() - start
            eta = (END - BASE)/((idx - BASE)/ttot)
            hrs = eta//3600
            mins = (eta - 3600*hrs)//60
            secs = (eta - 3600*hrs - 60*mins)

    ttot = time.time() - start
    hrs = ttot//3600
    mins = (ttot - 3600*hrs)//60
    secs = (ttot - 3600*hrs - 60*mins)
    channel_idx += 1 


dest = f'{PATH}/{DATE}.h5'

src = f'{PATH}/{DATE}_sfc.nc'
writetofile(src, dest, 0, 'u10') # u10
writetofile(src, dest, 1, 'v10') # v10
writetofile(src, dest, 2, 't2m') # t2m
writetofile(src, dest, 3, 'sp') # sp
writetofile(src, dest, 4, 'msl') #mslp

src = f'{PATH}/{DATE}_pl.nc'
writetofile(src, dest, 5, 't', 2) # t850

writetofile(src, dest, 6, 'u', 3) # u1000
writetofile(src, dest, 7, 'v', 3) # v1000
writetofile(src, dest, 8, 'z', 3) # z1000

writetofile(src, dest, 9, 'u', 2) # u850
writetofile(src, dest, 10, 'v', 2) # v850
writetofile(src, dest, 11, 'z', 2) # z850

writetofile(src, dest, 12, 'u', 1) # u500
writetofile(src, dest, 13, 'v', 1) # v500
writetofile(src, dest, 14, 'z', 1) # z500

writetofile(src, dest, 15, 't', 1) # t500
writetofile(src, dest, 16, 'z', 0) # z50

writetofile(src, dest, 17, 'r', 1) # r500
writetofile(src, dest, 18, 'r', 2) # r850

src = f'{PATH}/{DATE}_sfc.nc'
writetofile(src, dest, 19, 'tcwv') # tcwv
writetofile(src, dest, 20, 'sst') # sst
