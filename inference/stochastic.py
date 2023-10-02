# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation
# Filip Øskar Łanecki - Jagiellonian University, Cyfronet ACK

import os
import sys
import time
import logging
import argparse

import h5py
import numpy as np
import torch
import torch.distributed as dist

from collections import OrderedDict
from utils import logging_utils
from utils.YParams import YParams
from networks.afnonet import AFNONet

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
logging_utils.config_logger()


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)


def load_model(model, checkpoint_file):
    model.zero_grad()
    checkpoint = torch.load(checkpoint_file)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:  # don't know what can fail here, since python does not say
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x,
                                           scale_factor=scale,
                                           mode='bilinear')


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available(
    ) else 'cpu'

    # -- load the validation data
    file_path = params.data_path
    if params.log_to_screen:
        logging.info('loading data')
        logging.info('  data from {}'.format(file_path))
    with h5py.File(file_path, 'r') as f:
        data = torch.as_tensor(np.array(f['fields'])).to(device,
                                                         dtype=torch.float)

    params['N_in_channels'] = len(np.array(
        params.in_channels))  # necessary for the model
    params['N_out_channels'] = len(np.array(
        params.out_channels))  # necessary for the model

    # -- load the model
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)
    else:
        raise Exception("not implemented")

    checkpoint_file = params['best_checkpoint_path']
    if params.log_to_screen:
        logging.info(
            'loading trained model checkpoint from {}'.format(checkpoint_file))
    model = load_model(model, checkpoint_file).to(device)

    return data, model, device


def autoregressive_inference(params, data, model, device):
    # -- initialize global variables
    n_perturbations = int(params.n_pert)

    # -- initialize memory for image sequence
    seq_pred = torch.zeros(
        (data.shape[0] * n_perturbations, 1, data.shape[2], data.shape[3],
         data.shape[4])).to(device, dtype=torch.float)

    # -- autoregressive inference
    with torch.no_grad():
        for line in range(data.shape[0]):
            for pert in range(n_perturbations):
                history = data[line] if pert == 0 else gaussian_perturb(
                    data[line], level=params.n_level, device=device)
                future = model(history)
                seq_pred[line + pert * data.shape[0]] = future

    return seq_pred


if __name__ == '__main__':
    # -- prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config",
                        default='./config/AFNO.yaml',
                        type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--data_path_input",
                        default=None,
                        type=str,
                        help='path to data used for prediction')
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help='directory to store output data')
    parser.add_argument("--suffix",
                        default="",
                        type=str,
                        help='directory to store output data')
    parser.add_argument("--prediction_length", default=1, type=int)

    # -- parse config file and script args
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['data_path'] = args.data_path_input

    # -- prepare world size for multithreading
    params['world_size'] = int(
        os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    world_rank = 0
    world_size = params.world_size
    params['global_batch_size'] = params.batch_size

    local_rank = 0
    if params['world_size'] > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', init_method='env://')
        args.gpu = local_rank
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # set up directory
    expDir = args.output_dir

    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = params['weights_path']
    params['resuming'] = False
    params['local_rank'] = local_rank

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None,
                                  log_filename=os.path.join(
                                      expDir, 'inference_out.log'))
        logging_utils.log_versions()
        # params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    # -- get data and models
    data, model, device = setup(params)

    # -- actual prediction happens here
    logging.info("begining stochastic inference")
    prediction_length = int(args.prediction_length)

    seq_pred = data
    for step in range(prediction_length):
        t0 = time.time()
        seq_pred = autoregressive_inference(params, seq_pred, model, device)
        logging.info(
            f"time for inference at step {step + 1} = {time.time() - t0}")
    seq_pred = seq_pred.cpu().numpy()

    # -- save prediction
    h5name = os.path.join(
        params['experiment_dir'],
        'stochastic_autoregressive_prediction' + args.suffix + '.h5')

    if dist.is_initialized():
        if params.log_to_screen:
            logging.info("saving files at {}".format(h5name))
        dist.barrier()
        from mpi4py import MPI
        with h5py.File(h5name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
            if "fields" in f.keys():
                del f["fields"]
            f.create_dataset("fields",
                             data=seq_pred,
                             shape=seq_pred.shape,
                             dtype=np.float32)
        dist.barrier()
    else:
        if params.log_to_screen:
            logging.info("saving files at {}".format(h5name))
        with h5py.File(h5name, 'a') as f:
            if "fields" in f.keys():
                del f["fields"]
            f.create_dataset("fields",
                             data=seq_pred,
                             shape=seq_pred.shape,
                             dtype=np.float32)
