FROM nvcr.io/nvidia/pytorch:21.11-py3

# update repo info
RUN apt update -y

# install mpi4py
RUN pip install mpi4py

# h5py
RUN pip install h5py

# other python stuff
RUN pip install wandb && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install timm && \
    pip install einops

# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# set wandb to offline
#ENV WANDB_MODE offline

# copy source code
RUN mkdir -p /opt/ERA5_wind
COPY config /opt/ERA5_wind/config
COPY copernicus /opt/ERA5_wind/copernicus
COPY docker /opt/ERA5_wind/docker
COPY networks /opt/ERA5_wind/networks
COPY utils /opt/ERA5_wind/utils
COPY plotting /opt/ERA5_wind/plotting
COPY mpu /opt/ERA5_wind/mpu
COPY *.py /opt/ERA5_wind/
COPY *.sh /opt/ERA5_wind/
COPY perf_tests /opt/perf_tests

# create dummy git image
RUN cd /opt/ERA5_wind && git init
