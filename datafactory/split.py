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
chunk = int(data.shape[0] / args.splits)
for split in range(args.splits):
    file_name = f'{args.data_path_input[:-3]}_split_{split}.h5'
    new_data = data[split * chunk:(split + 1) * chunk]
    with h5py.File(file_name, 'a') as f:
        if HDF_DATASET in f.keys():
            del f[HDF_DATASET]
        f.create_dataset(HDF_DATASET,
                         data=new_data,
                         shape=new_data.shape,
                         dtype=np.float32)
