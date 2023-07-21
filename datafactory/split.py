import argparse
import h5py
import numpy as np

HDF_DATASET = 'fields'

# -- prepare argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--splits", type=int)
parser.add_argument("--data_path_input", type=str)

# -- parse config file and script args
args = parser.parse_args()

# -- get data and models
data = h5py.File(args.data_path_input, 'r')[HDF_DATASET]

# -- select data
# -- save standardized data
with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    # are we not saving the data twice here ??
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
    f[HDF_DATASET][...] = data
