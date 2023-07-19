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
    'reanalysis-era5-single-levels', {
        'product_type':'reanalysis',
        'format':'netcdf',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            'mean_sea_level_pressure',
            'surface_pressure',
            'total_column_water_vapour',
        ],
        'year':args.year,
        'month':args.month,
        'day':[args.day],
        'time':[args.hour],
    }, f'{args.datapath}/{args.filename}_sfc.nc')
