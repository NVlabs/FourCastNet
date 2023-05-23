import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
        ],
        'year': '{{= year }}',
        'month': '{{= month }}',
        'day': ['{{= day }}'],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    '{{= datapath }}/{{= filename }}_sfc.nc')
