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

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import h5py


def write_to_file(src, dest, src_name, src_index, dest_idx, mult_g=False):

    if mult_g:
        scale_factor = 9.8
    else:
        scale_factor = 1.0

    handle = xr.load_dataset(src, engine="pynio")    

    if len(handle[src_name].shape) == 3: 
        ds = handle[src_name][src_index]
    elif len(handle[src_name].shape) == 2: 
        ds = handle[src_name][:]

    with h5py.File(dest, 'a') as f:
        f['fields'][0, dest_idx] = scale_factor*ds 
        f.flush()


destdir = './h5files/'

if not os.path.exists(destdir):
    os.makedirs(destdir)

dest = destdir + '/init.h5'

with h5py.File(dest, 'w') as f:
    f.create_dataset('fields', dtype='f', shape=(1,26,721,1440))

file_ = './gribfiles/tuvzr.grb'
ds = xr.load_dataset(file_, engine="pynio")
print(ds['lv_ISBL0'])

#u10
src = './gribfiles/u10.grb' 
write_to_file(src, dest, 'UGRD_P0_L103_GLL0', 0, 0)

#v10
src = './gribfiles/v10.grb' 
write_to_file(src, dest, 'VGRD_P0_L103_GLL0', 0, 1)

#t2m
src = './gribfiles/t2m.grb' 
write_to_file(src, dest, 'TMP_P0_L103_GLL0', 0, 2)

#sp
src = './gribfiles/sp.grb' 
write_to_file(src, dest, 'PRES_P0_L1_GLL0', 0, 3)

#mslp
src = './gribfiles/mslp.grb' 
write_to_file(src, dest, 'PRMSL_P0_L101_GLL0', 0, 4)

#t850
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'TMP_P0_L100_GLL0', 3, 5)

#u1000
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'UGRD_P0_L100_GLL0', 4, 6)

#v1000
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'VGRD_P0_L100_GLL0', 4, 7)

#z1000
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'HGT_P0_L100_GLL0', 4, 8, mult_g=True)

#u850
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'UGRD_P0_L100_GLL0', 3, 9)

#v850
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'VGRD_P0_L100_GLL0', 3, 10)

#z850
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'HGT_P0_L100_GLL0', 3, 11, mult_g=True)

#u500
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'UGRD_P0_L100_GLL0', 2, 12)

#v500
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'VGRD_P0_L100_GLL0', 2, 13)

#z500
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'HGT_P0_L100_GLL0' , 2, 14, mult_g=True)

#t500
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'TMP_P0_L100_GLL0' , 2, 15)

#z50
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'HGT_P0_L100_GLL0', 0, 16, mult_g=True)

#r500
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'RH_P0_L100_GLL0', 2, 17)

#r850
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'RH_P0_L100_GLL0', 3, 18)

#tcwv
src = './gribfiles/tcwv.grb' 
write_to_file(src, dest, 'PWAT_P0_L200_GLL0', 0, 19)

#u100
src = './gribfiles/u100.grb' 
write_to_file(src, dest, 'UGRD_P0_L103_GLL0', 0, 20)

#v100
src = './gribfiles/v100.grb' 
write_to_file(src, dest, 'VGRD_P0_L103_GLL0', 0, 21)

#u250
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'UGRD_P0_L100_GLL0', 1, 22)

#v250
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'VGRD_P0_L100_GLL0', 1, 23)

#z250
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'HGT_P0_L100_GLL0', 1, 24, mult_g=True)

#t250
src = './gribfiles/tuvzr.grb' 
write_to_file(src, dest, 'TMP_P0_L100_GLL0', 1, 25)

