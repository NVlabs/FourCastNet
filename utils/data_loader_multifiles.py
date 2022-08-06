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

import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
#import cv2
from utils.img_utils import reshape_fields, reshape_precip


def get_data_loader(params, files_pattern, distributed, train):

  dataset = GetDataset(params, files_pattern, train)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size),
                          num_workers=params.num_data_workers,
                          shuffle=False, #(sampler is None),
                          sampler=sampler if train else None,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
  def __init__(self, params, location, train):
    self.params = params
    self.location = location
    self.train = train
    self.dt = params.dt
    self.n_history = params.n_history
    self.in_channels = np.array(params.in_channels)
    self.out_channels = np.array(params.out_channels)
    self.n_in_channels = len(self.in_channels)
    self.n_out_channels = len(self.out_channels)
    self.crop_size_x = params.crop_size_x
    self.crop_size_y = params.crop_size_y
    self.roll = params.roll
    self._get_files_stats()
    self.two_step_training = params.two_step_training
    self.orography = params.orography
    self.precip = True if "precip" in params else False
    self.add_noise = params.add_noise if train else False

    if self.precip:
        path = params.precip+'/train' if train else params.precip+'/test'
        self.precip_paths = glob.glob(path + "/*.h5")
        self.precip_paths.sort()

    try:
        self.normalize = params.normalize
    except:
        self.normalize = True #by default turn on normalization if not specified in config

    if self.orography:
      self.orography_path = params.orography_path

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.location + "/*.h5")
    self.files_paths.sort()
    self.n_years = len(self.files_paths)
    with h5py.File(self.files_paths[0], 'r') as _f:
        logging.info("Getting file stats from {}".format(self.files_paths[0]))
        self.n_samples_per_year = _f['fields'].shape[0]
        #original image shape (before padding)
        self.img_shape_x = _f['fields'].shape[2] -1#just get rid of one of the pixels
        self.img_shape_y = _f['fields'].shape[3]

    self.n_samples_total = self.n_years * self.n_samples_per_year
    self.files = [None for _ in range(self.n_years)]
    self.precip_files = [None for _ in range(self.n_years)]
    logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
    logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
    logging.info("Delta t: {} hours".format(6*self.dt))
    logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


  def _open_file(self, year_idx):
    _file = h5py.File(self.files_paths[year_idx], 'r')
    self.files[year_idx] = _file['fields']  
    if self.orography:
      _orog_file = h5py.File(self.orography_path, 'r')
      self.orography_field = _orog_file['orog']
    if self.precip:
      self.precip_files[year_idx] = h5py.File(self.precip_paths[year_idx], 'r')['tp']
    
  
  def __len__(self):
    return self.n_samples_total


  def __getitem__(self, global_idx):
    year_idx = int(global_idx/self.n_samples_per_year) #which year we are on
    local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

    y_roll = np.random.randint(0, 1440) if self.train else 0#roll image in y direction

    #open image file
    if self.files[year_idx] is None:
        self._open_file(year_idx)

    if not self.precip:
      #if we are not at least self.dt*n_history timesteps into the prediction
      if local_idx < self.dt*self.n_history:
          local_idx += self.dt*self.n_history

      #if we are on the last image in a year predict identity, else predict next timestep
      step = 0 if local_idx >= self.n_samples_per_year-self.dt else self.dt
    else:
      inp_local_idx = local_idx
      tar_local_idx = local_idx
      #if we are on the last image in a year predict identity, else predict next timestep
      step = 0 if tar_local_idx >= self.n_samples_per_year-self.dt else self.dt
      # first year has 2 missing samples in precip (they are first two time points)
      if year_idx == 0:
        lim = 1458
        local_idx = local_idx%lim 
        inp_local_idx = local_idx + 2
        tar_local_idx = local_idx
        step = 0 if tar_local_idx >= lim-self.dt else self.dt

    #if two_step_training flag is true then ensure that local_idx is not the last or last but one sample in a year
    if self.two_step_training:
        if local_idx >= self.n_samples_per_year - 2*self.dt:
            #set local_idx to last possible sample in a year that allows taking two steps forward
            local_idx = self.n_samples_per_year - 3*self.dt

    if self.train and self.roll:
      y_roll = random.randint(0, self.img_shape_y)
    else:
      y_roll = 0

    if self.orography:
        orog = self.orography_field[0:720] 
    else:
        orog = None

    if self.train and (self.crop_size_x or self.crop_size_y):
      rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
      rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
    else: 
      rnd_x = 0
      rnd_y = 0
      
    if self.precip:
      return reshape_fields(self.files[year_idx][inp_local_idx, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train), \
                reshape_precip(self.precip_files[year_idx][tar_local_idx+step], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)
    else:
        if self.two_step_training:
            return reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    reshape_fields(self.files[year_idx][local_idx + step:local_idx + step + 2, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)
        else:
            return reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    reshape_fields(self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)





    


  
    

