#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import glob
import sys
import os
import xarray as xr
import datetime
from cartopy import crs as ccrs
from tqdm import tqdm
import json
from netCDF4 import Dataset as DS

f = DS('./forecasts/autoregressive_predictions.nc', 'r')
print(f)

def make_gif(frame_folder, gif_path, gif_name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save(gif_path +  gif_name, format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)

def plot_field(field, scale, units, cmap, pred_title, savepath):

    lat = np.arange(-90,90,0.25)
    lat = lat[::-1]
    lon = np.arange(0,360,0.25)
    field = xr.DataArray(field, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
    vmin, vmax = scale
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_aspect('equal', adjustable='box')
    ax.coastlines()
    im = field.plot(ax=ax, vmax=vmax, vmin=vmin, cmap = cmap, cbar_kwargs={'label': units, 'orientation': 'vertical', 'fraction': 0.024, 'pad': 0.04, 'location': 'right'})
    ax.set_xticks(np.arange(-180, 180, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 90, 30), crs=ccrs.PlateCarree())
    ax.set_title(pred_title)
    ax.grid(True)
    fig.savefig(savepath + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def make_frames_speed(uch, vch, scale, cmap, date, field_name, savepath):
    if not os.path.isdir(savepath):
      os.makedirs(savepath)
    with h5py.File('./forecasts/autoregressive_predictions.nc', 'r') as f:
    
      u = f['predicted'][0,:,uch]
      v = f['predicted'][0,:,vch]
    
    initialization_date = datetime.datetime.strptime(date, '%Y-%m-%d-%H')

    print("plotting frames for {}".format(field_name))
    for t in tqdm(range(1,24)):
      forecast_date = initialization_date + datetime.timedelta(hours=t*6)
      title = field_name + ' forecast using FourCastNet ' +  ' initialized at: ' + initialization_date.strftime('%Y-%m-%d %H:%M') + '\n Forecast: ' + forecast_date.strftime('%Y-%m-%d %H:%M')
      plot_field(np.sqrt(u[t]**2 + v[t]**2), scale, "m/s", cmap, title, savepath + str(t).zfill(4)) 

def make_frames(ch, scale, cmap, units, date, field_name, savepath):
    if not os.path.isdir(savepath):
      os.makedirs(savepath)
    with h5py.File('./forecasts/autoregressive_predictions.nc', 'r') as f:

      var = f['predicted'][0,:,ch]

    initialization_date = datetime.datetime.strptime(date, '%Y-%m-%d-%H')
    print("plotting frames for {}".format(field_name))
    for t in tqdm(range(1,24)):
      
      forecast_date = initialization_date + datetime.timedelta(hours=t*6)
      title = field_name + ' forecast using FourCastNet ' +  ' initialized at: ' + initialization_date.strftime('%Y-%m-%d %H:%M') + '\n Forecast: ' + forecast_date.strftime('%Y-%m-%d %H:%M')
      plot_field(var[t], scale, units, cmap, title, savepath + str(t).zfill(4)) 



date = sys.argv[1] 

#read the netcdf file
with h5py.File('./forecasts/autoregressive_predictions.nc', 'r') as f:
  channel_short_names = list(f.attrs['channel_short_names'][:])
  channel_descriptions = list(f.attrs['channel_descriptions'][:])

for i in range(len(channel_short_names)):
  print("{} ---- {}".format(i, channel_descriptions[i]))


frames_path = './images/sfc-speed/'
uidx = channel_short_names.index('u10')
vidx = channel_short_names.index('v10')
make_frames_speed(uidx, vidx, (0,20), 'RdBu_r', date, 'surface wind speed', savepath = frames_path)
make_gif(frames_path, frames_path + '/gifs/','sfc-speed.gif')

frames_path = './images/250hpa-speed/'
uidx = channel_short_names.index('u250')
vidx = channel_short_names.index('v250')
make_frames_speed(uidx,vidx, (0, 60), 'RdBu_r', date, '250hPa wind speed', savepath = frames_path)
make_gif(frames_path, frames_path + '/gifs/','250-speed.gif')

frames_path = './images/850hpa-speed/'
uidx = channel_short_names.index('u850')
vidx = channel_short_names.index('v850')
make_frames_speed(uidx,vidx, (0, 30), 'RdBu_r', date, '850hPa wind speed', savepath = frames_path)
make_gif(frames_path, frames_path + '/gifs/','850-speed.gif')

frames_path = './images/tcwv/'
chidx = channel_short_names.index('tcwv')
make_frames(chidx, (0,65), 'viridis', "kg/m^2", date, 'TCVW', frames_path)
if not os.path.isdir(frames_path + '/gifs/'):
  os.makedirs(frames_path + '/gifs/')
make_gif(frames_path, frames_path + '/gifs/','tcwv.gif')

frames_path = './images/t2m/'
chidx = channel_short_names.index('t2m')
make_frames(chidx, (273-60,273+60), 'magma', "K", date, '2m Temperature', frames_path)
if not os.path.isdir(frames_path + '/gifs/'):
  os.makedirs(frames_path + '/gifs/')
make_gif(frames_path, frames_path + '/gifs/','t2m.gif')

frames_path = './images/z500/'
chidx = channel_short_names.index('z500')
make_frames(chidx, (50000,58000), 'plasma', "m^2/s^2", date, '500hPa geopotential height', frames_path)
if not os.path.isdir(frames_path + '/gifs/'):
  os.makedirs(frames_path + '/gifs/')
make_gif(frames_path, frames_path + '/gifs/','z500.gif')


