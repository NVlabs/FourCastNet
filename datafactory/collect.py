import argparse
import h5py
import numpy as np

HDF_DATASET = 'fields'

# -- prepare argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--splits", type=int)
parser.add_argument("--data_path_prefix", type=str)
parser.add_argument("--data_path_output", type=str)

# -- parse config file and script args
args = parser.parse_args()

# -- get data and models
datas = [
    h5py.File(f'{args.data_path_prefix}_split_{split}.h5', 'r')[HDF_DATASET]
    for split in range(args.splits)
]
data = np.concatenate(datas, axis=0)

with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
