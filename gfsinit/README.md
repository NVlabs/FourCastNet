This code generates real-time AI forecasts using FourCastNet (FCN) by initializing a trained FCN model using real-time GFS analysis states from the NOAA NCEP production server. Instructions for running this code can be found in this wiki after the visualizations.



![sfc_speed](./images/sfc-speed/gifs/sfc-speed.gif)

![TCWV](./images/tcwv/gifs/tcwv.gif)

![t2m](./images/t2m/gifs/t2m.gif)

![z500](./images/z500/gifs/z500.gif)

![850-speed](./images/850hpa-speed/gifs/850-speed.gif)

![250-speed](./images/250hpa-speed/gifs/250-speed.gif)

## Instructions:

We recommend using a docker container to run this code. The dockerfile for building the container is provided in the ```docker``` directory. 

generate a forecast and visualize using ```./run.sh```

