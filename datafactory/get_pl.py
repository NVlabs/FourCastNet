import cdsapi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=str)
parser.add_argument("--month", type=str)
parser.add_argument("--day", type=str)
parser.add_argument("--hour", type=str)
parser.add_argument("--datapath", type=str)
parser.add_argument("--filename", type=str)

args = parser.parse_args()
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels', {
        'product_type':'reanalysis',
        'format':'netcdf',
        'variable': [
            'geopotential',
            'relative_humidity',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
        ],
        'pressure_level': [
            '50',
            '500',
            '850',
            '1000',
        ],
        'year':args.year,
        'month':args.month,
        'day':[args.day],
        'time':[args.hour],
    }, f'{args.datapath}/{args.filename}_pl.nc')
