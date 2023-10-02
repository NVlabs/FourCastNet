import argparse
import h5py
import numpy as np
import gudhi
from scipy.spatial.distance import cdist

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

# -- prepare for symplexing
target = datas[0].shape[0]
data_safe = np.concatenate(datas, axis=0)
data = data_safe.reshape(
    (data_safe.shape[0],
     data_safe.shape[2] * data_safe.shape[3] * data_safe.shape[4]))

distance_matrix = cdist(data, data)
minimum = np.min(distance_matrix[np.nonzero(distance_matrix)])
maximum = np.max(distance_matrix[np.nonzero(distance_matrix)])
target_distance = minimum + 0.01
step = (maximum - minimum) / (data.shape[0] * 16)

# -- build symplex tree
while True:
    cplx = gudhi.SimplexTree.create_from_array(distance_matrix,
                                               max_filtration=target_distance)
    cplx.expansion(1)
    cplx.compute_persistence(homology_coeff_field=2)
    if cplx.betti_numbers()[0] <= target:
        break
    else:
        target_distance += step

# -- clusterize
vertices = np.arange(data.shape[0])
components = []
for vertex in vertices:
    anycomp = False
    for component in components:
        found = False
        for u in component:
            if cplx.find([u, vertex]):
                found = True
                break
        if found:
            component.append(vertex)
            anycomp = True
            break
    if not anycomp:
        components.append([vertex])

data = np.array([
    np.array([data_safe[j] for j in component]).sum(axis=0) / len(component)
    for component in components
])

with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
