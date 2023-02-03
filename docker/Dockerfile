FROM nvcr.io/nvidia/pytorch:22.07-py3
# update repo info
RUN apt update -y

RUN conda create -n grib xarray pynio matplotlib h5py cartopy netcdf4 tqdm

# install mpi4py
RUN pip install mpi4py

# h5py
RUN pip install h5py

# other python stuff
RUN pip install wandb && \ 
    pip install ruamel.yaml && \ 
    pip install --upgrade tqdm && \ 
    pip install timm && \ 
    pip install einops && \ 
    pip install netcdf4


RUN apt-get install git-lfs