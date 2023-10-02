import argparse
import h5py
import numpy as np
import gudhi
from scipy.spatial.distance import cdist

HDF_DATASET = 'fields'
STEP = 0.1

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
target = datas[0].shape[0]
data = np.concatenate(datas, axis=0)

distance_matrix = cdist(data, data)
target_distance = 0.0

while True:
    cplx = gudhi.SimplexTree.create_from_array(distance_matrix,
                                               max_filtration=target_distance)
    cplx.expansion(1)
    cplx.compute_persistence(homology_coeff_field=2)
    if cplx.betti_numbers()[0] < target:
        break
    else:
        target_distance += STEP

# HERE SHOULD CLUSTERIZE

with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
