import argparse
import h5py
import numpy as np

HDF_DATASET = 'fields'

# -- prepare argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--month", type=int)
parser.add_argument("--day", type=int)
parser.add_argument("--hour", type=str)
parser.add_argument("--data_path_output", type=str)

# -- parse config file and script args
args = parser.parse_args()

# -- get data and models
data = h5py.File('/net/pr2/projects/plgrid/plggorheuro/fourcast/data/2018.h5',
                 'r')[HDF_DATASET]


# -- select data
def days_in_month(month):
    return 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30


def index_from_date(hour, day, month):
    hours_map = {'00:00': 0, '06:00': 1, '12:00': 2, '18:00': 3}
    index = 0
    for m in range(1, month):
        index += 4 * days_in_month(m)
    for _d in range(1, day):
        index += 4
    for _h in range(hours_map[hour]):
        index += 1
    return index


ix = index_from_date(args.hour, args.day, args.month)
data = data[ix:ix + 1]

# -- save standardized data
with h5py.File(args.data_path_output, 'a') as f:
    if HDF_DATASET in f.keys():
        del f[HDF_DATASET]
    f.create_dataset(HDF_DATASET,
                     data=data,
                     shape=data.shape,
                     dtype=np.float32)
