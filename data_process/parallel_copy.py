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
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

def writetofile(src, dest, channel_idx, varslist):
    if os.path.isfile(src):
        batch = 2**6
        rank = MPI.COMM_WORLD.rank
        Nproc = MPI.COMM_WORLD.size
        Nimgtot = 1460#src_shape[0]

        Nimg = Nimgtot//Nproc
        print("Nimgtot",Nimgtot)
        print("Nproc",Nproc)
        print("Nimg",Nimg)
        base = rank*Nimg
        end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
        idx = base

        for variable_name in varslist:

            fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

            start = time.time()
            while idx<end:
                if end - idx < batch:
                    ims = fsrc[idx:end]
                    print(ims.shape)
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    break
                else:
                    ims = fsrc[idx:idx+batch]
                    fdest['fields'][idx:idx+batch, channel_idx, :, :] = ims
                    idx+=batch
                    ttot = time.time() - start
                    eta = (end - base)/((idx - base)/ttot)
                    hrs = eta//3600
                    mins = (eta - 3600*hrs)//60
                    secs = (eta - 3600*hrs - 60*mins)

            ttot = time.time() - start
            hrs = ttot//3600
            mins = (ttot - 3600*hrs)//60
            secs = (ttot - 3600*hrs - 60*mins)
            channel_idx += 1 

#year_dict = {'j': np.arange(1979, 1993), 'k': np.arange(1993, 2006), 'a' : np.arange(2006, 2021)}

dir_dict = {}
for year in np.arange(1979, 1993):
    dir_dict[year] = 'j'

for year in np.arange(1993, 2006):
    dir_dict[year] = 'k'

for year in np.arange(2006, 2021):
    dir_dict[year] = 'a'


print(dir_dict)

years = np.arange(1979, 2018)
    
for year in years:

    print(year)
    src = '/global/cscratch1/sd/jpathak/ERA5_u10v10t2m_netcdf/10m_u_v_2m_t_gp_lsm_toa_' + str(year) + '.nc'
    dest = '/global/cscratch1/sd/jpathak/ERA5/wind/vlevels/' + str(year) + '.h5'
    writetofile(src, dest, 0, ['u'])
    writetofile(src, dest, 1, ['v'])

    src = '/project/projectdirs/dasrepo/jpathak/ERA5_more_data/z_u_v_1000_'+str(year)+'.nc'
    writetofile(src, dest, 2, ['u'])
    writetofile(src, dest, 3, ['v'])
    writetofile(src, dest, 4, ['z'])

    usr = dir_dict[year]
    src ='/project/projectdirs/dasrepo/ERA5/wind_levels/6hr/' + usr +  '/u_v_z_pressure_level_850_' +str(year) + '.nc'
    writetofile(src, dest, 5, ['u'])
    writetofile(src, dest, 6, ['v'])
    writetofile(src, dest, 7, ['z'])

    usr = dir_dict[year]
    src ='/project/projectdirs/dasrepo/ERA5/wind_levels/6hr/' + usr +  '/u_v_z_pressure_level_500_' +str(year) + '.nc'
    writetofile(src, dest, 8, ['u'])
    writetofile(src, dest, 9, ['v'])
    writetofile(src, dest, 10, ['z'])

    src = '/project/projectdirs/dasrepo/jpathak/ERA5_more_data/z50_'+str(year)+'.nc'
    writetofile(src, dest, 11, ['z'])


