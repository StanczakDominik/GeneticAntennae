import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

population = gpd.GeoDataFrame.from_csv('GEOSTAT_grid_POP_1K_2011_V2_0_1.csv')

location_strings = population['GRD_ID'].values
coordinates_arr = np.zeros((len(location_strings), 2))
for i, loc in enumerate(location_strings):
    first_part, coordinates = loc.split('N')
    N, E = coordinates.split('E')
    coordinates_arr[i] = float(N)/100, float(E)/100
populations = population.index.values #in 1k units
countries = population['CNTR_CODE'].values

df = pd.DataFrame()
df['N'] = coordinates_arr[:,0]
df['E'] = coordinates_arr[:,1]
df['populations'] = populations
df['countries'] = countries
df.to_csv("fixed_data.csv", index=True)


df = pd.read_csv('fixed_data.csv', index_col=0)
