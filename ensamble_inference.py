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
import time
import numpy as np
import argparse
import sys

import h5py
import torch
import torch.distributed as dist
from collections import OrderedDict
import logging
from utils import logging_utils
# from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels

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
    except:
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
    valid_data_full = h5py.File(file_path, 'r')['fields']

    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(np.array(
        params.in_channels))  # necessary for the model
    params['N_out_channels'] = len(out_channels)  # necessary for the model
    params.means = np.load(params.global_means_path)[
        0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

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

    return valid_data_full, model


def standardize(data, params):
    return (data[:, np.array(params.in_channels), 0:720] -
            params.means) / params.stds


def autoregressive_inference(params, data, model):
    device = torch.cuda.current_device() if torch.cuda.is_available(
    ) else 'cpu'

    # -- initialize global variables
    n_perturbations = int(params.n_pert)
    prediction_length = int(params.prediction_length)

    data = torch.as_tensor(data).to(device, dtype=torch.float)

    # -- initialize memory for image sequence
    seq_pred = torch.zeros((n_perturbations, 1, data.shape[2], data.shape[3],
                            data.shape[4])).to(device, dtype=torch.float)

    # -- perturb data
    seq_pred[0] = data[0]
    for pert in range(1, n_perturbations):
        seq_pred[pert] = gaussian_perturb(data[0],
                                          level=params.n_level,
                                          device=device)

    # -- autoregressive inference
    with torch.no_grad():
        for step in range(prediction_length):
            for pert in range(n_perturbations):
                future = model(seq_pred[pert])
                seq_pred[pert] = future

    return seq_pred.cpu().numpy()


if __name__ == '__main__':
    # -- prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config",
                        default='./config/AFNO.yaml',
                        type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir",
                        default=None,
                        type=str,
                        help='path to store inference outputs')
    parser.add_argument("--weights",
                        default=None,
                        type=str,
                        help='path to model weights')
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        help='path to data used for prediction')
    parser.add_argument("--n_pert", default=3, type=int)
    parser.add_argument("--n_level", default=0.3, type=float)

    # -- parse config file and script args
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['n_pert'] = args.n_pert
    params['n_level'] = args.n_level
    params['data_path'] = args.data_path

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
    assert args.override_dir is not None, 'must set --override_dir argument'
    assert args.weights is not None, 'must set --weights argument'
    expDir = args.override_dir

    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights
    params['resuming'] = False
    params['local_rank'] = local_rank

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None,
                                  log_filename=os.path.join(
                                      expDir, 'inference_out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    # -- set filetag
    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:
        autoregressive_inference_filetag = ""
    autoregressive_inference_filetag += "_test"

    # -- get data and models
    valid_data_full, model = setup(params)

    # -- standardize input data
    standardized_data = standardize(valid_data_full[0:4], params)

    # -- actual prediction happens here
    logging.info("begining stochastic inference")

    seq_pred = np.expand_dims(standardized_data[-1:], 0)
    t0 = time.time()
    seq_pred = autoregressive_inference(params, seq_pred, model)
    logging.info(f"time for inference = {time.time() - t0}")

    # -- save prediction
    h5name = os.path.join(
        params['experiment_dir'], 'ensamble_autoregressive_predictions' +
        autoregressive_inference_filetag + '.h5')

    if params.log_to_screen:
        logging.info("saving files at {}".format(h5name))
    with h5py.File(h5name, 'a') as f:
        if "fields" in f.keys():
            del f["fields"]
        f.create_dataset("fields",
                         data=seq_pred,
                         shape=seq_pred.shape,
                         dtype=np.float32)
        f["fields"][...] = seq_pred
