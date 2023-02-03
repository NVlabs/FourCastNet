import os
import time
import numpy as np
import argparse

import h5py
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from networks.afnonet import AFNONet 
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from netCDF4 import Dataset as DS
import json


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def autoregressive_inference(params, ic):
    ic = int(ic)
    if dist.is_initialized():
        world_size = dist.get_world_size()
        print(world_size)
    else:
        world_size = 1

    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    forecast_dir = params['forecast_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = 720    
    img_shape_y = 1440
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    if params["orography"]:
      params['N_in_channels'] = n_in_channels + 1
    else:
      params['N_in_channels'] = n_in_channels

    params['N_out_channels'] = n_out_channels
    means = np.load(params.global_means_path)[0, out_channels]
    stds = np.load(params.global_stds_path)[0, out_channels]
    n_grid_channels = params.N_grid_channels
    orography = params.orography
    orography_path = params.orography_path
    #print(means.shape, stds.shape)
    
    train = False
    
    #load Model weights
    
    if params.log_to_screen:
      logging.info('Loading trained model checkpoint from {}'.format(params['checkpoint_path']))

    if params.nettype == 'afno':
      model = AFNONet(params).to(device) 
    else:
      raise Exception('Not implemented')



    model.zero_grad()
    checkpoint_file  = params['checkpoint_path']
    checkpoint = torch.load(checkpoint_file)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key.replace("module.", "")
            if name != 'ged':
                new_state_dict[name] = val 
        
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    seq_pred = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))

    #autoregressive inference
    if params.log_to_screen:
      logging.info('Begin autoregressive inference')

    if orography:
      orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0)).to(device, dtype = torch.float)
      print("orography loaded; shape:", orog.shape)

    init_path = params.inf_data_path

    with h5py.File(init_path, 'r') as f:
      gfs_init = f['fields'][0:1,:,0:720,:]
      gfs_init -= means[:,0:n_in_channels]
      gfs_init /= stds[:,0:n_in_channels]


    with torch.no_grad():
      for i in range(params.prediction_length): 
        
        if i==0: #start of sequence
          logging.info('Starting prediction')
          first = torch.as_tensor(gfs_init).to(device, dtype =torch.float)
          seq_pred[i] = first
          
          if params.perturb:
              first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
          else:
              future_pred = model(first)

        else:
          future_pred = model(future_pred)

        if i < prediction_length-1: #not on the last step

          logging.info('Step {} of {} - forecast lead time {} hours'.format(i+1, prediction_length, (i+1)*6))
          
          seq_pred[i+1] = future_pred
          
    return np.expand_dims(seq_pred, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default = 0, type = float)
    parser.add_argument("--override_dir", default = 'None', type = str)
    parser.add_argument("--checkpoint", default = 'None', type = str)
    

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
      params['world_size'] = int(os.environ['WORLD_SIZE'])

    world_rank = 0
    if params['world_size'] > 1:
      torch.cuda.set_device(args.local_rank)
      dist.init_process_group(backend='nccl',
                              init_method='env://')
      args.gpu = args.local_rank
      world_rank = dist.get_rank()
      params['global_batch_size'] = params.batch_size
      params['batch_size'] = int(params.batch_size//params['world_size'])

    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir =='None':
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    else:
      expDir = args.override_dir

    if  world_rank==0:
      if not os.path.isdir(expDir):
        os.makedirs(expDir)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    args.resuming = False

    params['resuming'] = args.resuming
    params['local_rank'] = args.local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = args.config
    if world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
      logging_utils.log_versions()
      params.log()

    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']


    n_ics = 1

    ics = [0]

    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:
        autoregressive_inference_filetag = ""


    #run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
      logging.info("Initial condition {} of {}".format(i+1, n_ics))
      sp = autoregressive_inference(params, ic)
      if i ==0:
        seq_pred = sp
      else:
        seq_pred = np.concatenate((seq_pred, sp), 0)
       
    prediction_length = seq_pred[0].shape[0]
    n_out_channels = seq_pred[0].shape[1]
    img_shape_x = seq_pred[0].shape[2]
    img_shape_y = seq_pred[0].shape[3]

    # re-scale seq_pred

    stds = np.load(params['global_stds_path'])
    means = np.load(params['global_means_path'])

    seq_pred *= stds[:,0:n_out_channels]
    seq_pred += means[:,0:n_out_channels]


    #save predictions and loss
    if params.log_to_screen:
      logging.info("Saving files at {}".format(os.path.join(params['forecast_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['forecast_dir'], 'autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:
      try:
        f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
      except:
        del f["predicted"]
        f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        f["predicted"][...]= seq_pred
    
