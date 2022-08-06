import cdsapi
import numpy as np
import os

usr = 'j' # j,k,a
base_path = '/project/projectdirs/dasrepo/ERA5/wind_levels/6hr/' + usr
if not os.path.isdir(base_path):
    os.makedirs(base_path)
year_dict = {'j': np.arange(1981, 1991), 'k': np.arange(1993,2006), 'a' : np.arange(2006, 2021)}
years = year_dict[usr]  
#t1 = [str(jj).zfill(2) for jj in range(1,4)] 
#t2 = [str(jj).zfill(2) for jj in range(4,7)] 
#t3 = [str(jj).zfill(2) for jj in range(7,10)] 
#t4 = [str(jj).zfill(2) for jj in range(10,13)] 
#
#trimesters = [t1, t2, t3, t4]
#months = [str(jj).zfill(2) for jj in range(1,13)] 
 
pressure_levels = [300]
c = cdsapi.Client()

for pressure_level in pressure_levels:
    
    for year in years:

        
        year_str = str(year) 
        pressure_str = str(pressure_level)
        file_str = base_path + '/u_v_z_pressure_level_'+ pressure_str + '_'  + year_str  + '.nc'
        print(year_str)
        print(file_str)
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'pressure_level': pressure_str,
                'variable': [
                    'u_component_of_wind', 'v_component_of_wind', 'geopotential',
                ],          
                'year': year_str,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00','18:00',
                ],          
            },
            file_str)
