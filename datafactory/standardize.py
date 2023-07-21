import os
import argparse
import h5py
import numpy as np
from utils.YParams import YParams

HDF_DATASET = 'fields'

# -- prepare argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
parser.add_argument("--config", default='full_field', type=str)
parser.add_argument("--data_path_input", type=str)
parser.add_argument("--data_path_output", type=str)

# -- parse config file and script args
args = parser.parse_args()
params = YParams(os.path.abspath(args.yaml_config), args.config)

# -- get data and models
data = h5py.File(args.data_path_input, 'r')[HDF_DATASET]

out_channels = np.array(params.out_channels)
means = np.load(params.global_means_path)[0, out_channels]
stds = np.load(params.global_stds_path)[0, out_channels]

# -- standardize input data
data = (data[:, np.array(params.in_channels), 0:720] - means) / stds
data = np.expand_dims(data[-1:], 0)

# -- save standardized data
with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
