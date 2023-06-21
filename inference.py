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
#Filip Øskar Łanecki - Jagiellonian University, Cyfronet ACK

import os
import time
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels
logging_utils.config_logger()
from utils.YParams import YParams
# from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet 
import wandb
import matplotlib.pyplot as plt
import glob

# -- i don't really understand this part, seems like this should be in the config
fld = "z500" # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 36 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10":0, "z500":14, "tp":0}


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    # -- this function does not seem to use params at all?
    model.zero_grad()
    checkpoint = torch.load(checkpoint_file)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')

def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # -- load the validation data
    file_path = params.data_path
    if params.log_to_screen:
        logging.info('loading data')
        logging.info('  data from {}'.format(file_path))
    valid_data_full = h5py.File(file_path, 'r')['fields']

    params.img_shape_x = valid_data_full.shape[2]
    params.img_shape_y = valid_data_full.shape[3]

    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(np.array(params.in_channels))
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # -- load the model
    if params.nettype == 'afno':
        model = AFNONet(params).to(device) 
    else:
      raise Exception("not implemented")
    checkpoint_file  = params['best_checkpoint_path']
    if params.log_to_screen:
        logging.info('loading trained model checkpoint from {}'.format(checkpoint_file))
    model = load_model(model, params, checkpoint_file).to(device)

    return valid_data_full, model

def autoregressive_inference(params, ic, valid_data_full, model): 
    ic = int(ic) 

    # -- initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    n_pert = params.n_pert

    print("enter auroregpred")
    print(f"nhistory: {n_history}")
    print(f"pred len: {prediction_length}")

    # -- initialize memory for image sequence
    seq_pred = torch.zeros((prediction_length+n_history, n_in_channels, img_shape_x-1, img_shape_y)).to(device, dtype=torch.float)

    # valid_data = valid_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] # -- extract valid data from first year
    valid_data = valid_data_full[0:8, in_channels, 0:720]
    # -- standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    true_vals = valid_data
    valid_data = valid_data[0:4]
    # load time means -- although, are they used anywhere later ?!
    # -- they do not get passed do the model, although i would not be surprised if it uses them in some other roundabout way
    # -- due to this stupid language not enforcing stuff
    # -- or maybe they were used in the rms calculation and i just forgot to delete them
    # m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:img_shape_x] # -- climatology
    # m = torch.unsqueeze(m, 0).to(device)
#    # m = m.to(device)
    # std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

    # -- autoregressive inference
    if params.log_to_screen:
      logging.info('begin autoregressive inference')
    
    with torch.no_grad():
        first = valid_data
        # for h in range(first.shape[0]):
        #     seq_pred[h] = first[h]
        # if params.perturb:
        #     first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
        future_pred = model(first)
        for j in range(4):
            print(f'first norm {torch.norm(first[j])}')
            print(f'future norm {torch.norm(future_pred[j])}')
        # -- the model returns an array of the same shape as was provided
        # -- but does it change anything inside? does it just always predict the same number of days as it is given?

      # print(valid_data.shape[0])
      # for i in range(valid_data.shape[0]): # -- what does this do? i have no idea
      #   if i == 0: # start of sequence
      #     first = valid_data[0:n_history+1]
      #     seq_pred = first
      #     if params.perturb:
      #       first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
      #     future_pred = model(first)
      #   else:
      #     future_pred = model(future_pred) # autoregressive step

      #   if i < prediction_length-1: # not on the last step
      #     seq_pred[n_history+i+1] = future_pred
      
    # if params.log_to_screen:
    #       logging.info('Predicted timestep {} of {}. {}'.format(i, prediction_length, fld))
    # print(f'seq_pred shape after {seq_pred.shape}')
    seq_pred[0:4] = first
    seq_pred[4:8] = future_pred

    print("===\nvalue comparison +4 to +1")
    for j in range(4):
        print(torch.norm(seq_pred[j+4]-true_vals[j+1]))
    print("===")

    # -- this shows that probably the whole tensor is shifted by one time step
    # -- so puting in times t0 to tn-1 gives times t1 to tn
    # -- we should test whether using the historical data gives better or worse results than simply passing in one time

    print("===\nvalue comparison +0 to +0")
    for j in range(4):
        print(torch.norm(seq_pred[j]-true_vals[j]))
    print("===")

    return seq_pred.cpu().numpy()

if __name__ == '__main__':
    # -- prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--n_pert", default=100, type=int)
    parser.add_argument("--n_level", default=0.3, type=float) # -- maybe we will be able to get rid of this
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--perturb", default=False, type=bool)
    
    # -- parse config file and script args
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['n_pert'] = args.n_pert
    params['perturb'] = args.perturb
    params['data_path'] = args.data_path

    # -- prepare world size for multithreading
    params['world_size'] = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    world_rank = 0
    world_size = params.world_size
    params['global_batch_size'] = params.batch_size

    local_rank = 0
    if params['world_size'] > 1:
      local_rank = int(os.environ["LOCAL_RANK"])
      dist.init_process_group(backend='nccl',
                              init_method='env://')
      args.gpu = local_rank
      world_rank = dist.get_rank()
      world_size = dist.get_world_size()
      params['global_batch_size'] = params.batch_size
      # params['batch_size'] = int(params.batch_size//params['world_size'])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir' 
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if world_rank==0:
      if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = local_rank

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = args.config

    if world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
      logging_utils.log_versions()
      params.log()

    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

    if fld == "z500" or fld == "t850":
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460

    # -- this ics shenanigans is real weird and needs to be figured out
    n_ics = params['n_initial_conditions']
    # -- this will only ever be default in our case
    # -- the following if basically overrides the parameter set above
    if params["ics_type"] == 'default':
        num_samples = n_samples_per_year-params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        n_ics = len(ics)
        logging.info("Inference for {} initial conditions".format(n_ics))

    # -- set filetag
    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:
        autoregressive_inference_filetag = ""
    params.n_level = args.n_level
    # autoregressive_inference_filetag += "_" + str(params.n_level) + "_" + str(params.n_pert) + "ens_" + fld
    autoregressive_inference_filetag += "_test"

    # -- get data and models
    valid_data_full, model = setup(params)

    # -- the following lines can be useful when we will want to parallelize splitting
    # run autoregressive inference for multiple initial conditions
    # parallelize over initial conditions -- but does it really?
    # if world_size > 1:
    #   tot_ics = len(ics)
    #   ics_per_proc = n_ics//world_size
    #   ics = ics[ics_per_proc*world_rank:ics_per_proc*(world_rank+1)] if world_rank < world_size - 1 else ics[(world_size - 1)*ics_per_proc:]
    #   n_ics = len(ics)
    #   logging.info('Rank %d running ics %s'%(world_rank, str(ics)))

    ics = [0] # -- to hell with ics, let me see what will happen
    n_ics = len(ics)
    # seq_pred = valid_data_full[0:4,:,0:720]
    for i, ic in enumerate(ics):
      t0 = time.time()
      logging.info("Initial condition {} of {}".format(i+1, n_ics))
      seq_pred = autoregressive_inference(params, ic, valid_data_full[0:8,:,0:720], model)
      print("===")
      print('seq pred norms:')
      for j in range(seq_pred.shape[0]):
        print(np.linalg.norm(seq_pred[j]))
      print(seq_pred.shape)
      print('===')
      # sp = autoregressive_inference(params, ic, valid_data_full, model)[:,:,0:720]
      # seq_pred = sp if i == 0 else np.concatenate((seq_pred, sp), 0)
      logging.info("Time for inference for ic {} = {}".format(i, time.time() - t0))
      params.n_history += 1


    prediction_length = seq_pred.shape[0]
    n_out_channels = seq_pred.shape[1]
    img_shape_x = seq_pred.shape[2]
    img_shape_y = seq_pred.shape[3]

    # -- save prediction
    h5name = os.path.join(params['experiment_dir'], 'ens_autoregressive_predictions' + autoregressive_inference_filetag + '.h5')

    # -- it appears that the only thing which is parallelized is actualy just saving the results to the h5 file
    if dist.is_initialized(): 
      if params.log_to_screen:
        logging.info("saving files at {}".format(h5name))
        logging.info("  array shapes: %s"%str((tot_ics, prediction_length, n_out_channels)))

      dist.barrier()
      from mpi4py import MPI
      with h5py.File(h5name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
        if "fields" in f.keys():
            del f["fields"]
        f.create_dataset("fields", data = seq_pred, shape = (prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        start = world_rank*ics_per_proc
        f["fields"][start:start+n_ics] = seq_pred
      dist.barrier()
    else:
      if params.log_to_screen:
        logging.info("saving files at {}".format(h5name))
      with h5py.File(h5name, 'a') as f:
        if "fields" in f.keys():
            del f["fields"]
        f.create_dataset("fields", data = seq_pred, shape = (prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        f["fields"][...] = seq_pred
