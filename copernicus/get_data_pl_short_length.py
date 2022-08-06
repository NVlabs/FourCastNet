import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '50', '500', '850',
            '1000',
        ],
        'year': '2021',
        'month': '10',
        'day': [
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    '/project/projectdirs/dasrepo/ERA5/oct_2021_19_31_pl.nc')
