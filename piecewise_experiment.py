import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from xyzservices import providers

from calibrate_sattelite_data import piecewise_linear_squares, weighted_least_squares, fit_data, fit_data_least, \
    calculate_weights
from plot_utils import plot_piecewise, plot_line
import pyproj
import contextily as ctx

import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import spearmanr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)

ctx.tile._disk_cache = True

print("Pyproj dir", pyproj.datadir.get_data_dir())

print("PyProj: ", pyproj.CRS.from_epsg(4326))

data = pd.read_hdf('data/data.h5')
all_data = data
cluster1 = data[(data['Longitude'] < -54.95) & (data['Latitude'] > -3.5)]
cluster2 = data[(data['Longitude'] > -54.95) & (data['Latitude'] > -3.5)]
cluster3 = data[(data['Longitude'] < -54.5) & (-3.5 > data['Latitude']) & (data['Latitude'] > -4)]
cluster4 = data[(data['Longitude'] > -54.5) & (data['Longitude'] < -54.1)]
cluster5 = data[data['Longitude'] > -54.1]

bot_south_river = cluster5[(cluster5['Latitude'] < -12.078)]
bot_north_river = cluster5[(cluster5['Latitude'] > -12.078) & (cluster5['Latitude'] < -12)]
middle = cluster5[(cluster5['Latitude'] > -12) & (cluster5['Latitude'] < -11.85)]
top_south_river = cluster5[(cluster5['Latitude'] > -11.83) & (cluster5['Latitude'] < -11.73)]
top_north_river = cluster5[(cluster5['Latitude'] > -11.73)]

cluster5_parts = [bot_south_river, bot_north_river, middle, top_south_river, top_north_river]

print("Cluster5", len(cluster5))
print("Parts: ", len(bot_south_river), len(bot_north_river), len(middle), len(top_south_river), len(top_north_river))
print("Sum", len(bot_north_river) + len(bot_south_river) + len(middle) + len(top_south_river) + len(top_north_river))

print("Sum: ", len(cluster1) + len(cluster2) + len(cluster3) + len(cluster4) + len(cluster5))

data = cluster2

data['geometry'] = gpd.points_from_xy(data['Longitude'], data['Latitude'])
gdf = gpd.GeoDataFrame(data, geometry='geometry')
gdf.set_crs(pyproj.CRS.from_epsg(4326), inplace=True)

# gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

# Plot second map
gdf.plot(ax=ax, color='blue', linewidth=2, aspect='equal')

ctx.add_basemap(ax, zoom='auto', crs=gdf.crs, source=ctx.providers.OpenTopoMap)
# ctx.add_basemap(ax, zoom='auto', crs=gdf.crs, source=ctx.providers.Esri.WorldImagery)
# ctx.add_basemap(ax, zoom='auto', crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
# ax.set_xlim([-54.2, -53.8])
# ax.set_ylim([-12.5, -11.5])
ax.set_xlim(gdf.total_bounds[[0, 2]] + (-0.1, 0.1))
ax.set_ylim(gdf.total_bounds[[1, 3]] + (-0.1, 0.1))

ax.set_title("Map 2")

plt.gca().set_aspect('auto')

plt.show()


# plt.scatter(data['Longitude'], data['Latitude'])
# plt.plot()
# plt.show()


# gdf = gpd.GeoDataFrame(data, geometry='geometry')

# gdf.set_crs("EPSG:4326", inplace=True)

# data = data[data['Longitude'] >= -54.]

def plot_data(data, name='plot'):
    # data['sen_delta'] = np.abs(data['geolocation_sensitivity_a2'] - data['sensitivity'])
    # data = data[data['Latitude'] > -12.]
    # data = data[data['Latitude'] < -12.078]
    # data = data[(data['Latitude'] > -11.83)  & (data['Latitude'] < -11.73)]
    # data = data[(data['Latitude'] > -11.73)]
    # data = data[(data['geolocation_sensitivity_a2'] > 0.94) & (data['geolocation_sensitivity_a2'] < 0.95)]
    # data = data[(data['geolocation_sensitivity_a2'] > 0.99)]
    # data = data[(data['als_h'] > 1.5) & (data['als_h'] < 3)]
    # data = data[(data['als_h'] < 1.5) ]
    # data = data[(data['rh_98'] < 15)]
    # data = data[data['quality_flag'] == 1]
    # print(data.sort_values(by='Longitude', ascending=False).tail(30)[['selected_algorithm', 'geolocation_sensitivity_a2', 'Site', 'rx_assess_quality_flag', 'degrade_flag', 'quality_flag', 'BEAM', 'solar_elevation', 'sensitivity', 'rh_98', 'als_h', 'Latitude', 'Longitude']])
    # print(data.sort_values(by='Latitude', ascending=False).head(100)[
    #           ['sen_delta', 'selected_algorithm', 'geolocation_sensitivity_a2', 'Site', 'rx_assess_quality_flag',
    #            'degrade_flag', 'quality_flag', 'BEAM', 'solar_elevation', 'sensitivity', 'rh_98', 'rh_99', 'rh_100',
    #            'als_h', 'Latitude', 'Longitude']])
    # data = data[(data['Latitude'] < -12.078) | (data['Latitude'] > -12)]
    # data = data[(data['Latitude'] > -12.078) & (data['Latitude'] < -12)]

    X = data['rh_98'].to_numpy().reshape(-1, 1)

    Y = data['als_h'].to_numpy()

    correlation98 = np.corrcoef(np.asarray(X).ravel(), Y)[0, 1]
    correlation99 = np.corrcoef(np.asarray(data['rh_99'].to_numpy().reshape(-1, 1)).ravel(), Y)[0, 1]
    correlation100 = np.corrcoef(np.asarray(data['rh_100'].to_numpy().reshape(-1, 1)).ravel(), Y)[0, 1]

    print(f"Correlation 98 / 99 / 100: {correlation98 * 100:.2f} / {correlation99 * 100:.2f} / {correlation100 * 100:.2f}")
    s_correlation98, p_value = spearmanr(data['rh_98'], data['als_h'])
    s_correlation99, p_value = spearmanr(data['rh_99'], data['als_h'])
    s_correlation100, p_value = spearmanr(data['rh_100'], data['als_h'])
    print(f"Spearman 98 / 99 / 100: {s_correlation98 * 100:.2f} / {s_correlation99 * 100:.2f} / {s_correlation100 * 100:.2f}")

    # print("Points left: ", len(data))
    # result = piecewise_linear_squares(X, Y, exp_name='Piecewise', data=data)
    # plot_piecewise(result)

    # result = fit_data_least(data, exp_name=name)

    result = fit_data(data, small_tree=False, exp_name=name)
    print(str(result))
    # plot_line(result, 'geolocation_sensitivity_a2')
    return result

original_data = all_data.copy()
results = []

for _ in range(1000):
    data = original_data.copy()
    permutation_order = np.random.permutation(len(data))

    data['weights'] = calculate_weights(data, small_tree=False)
    data['weights'] = data['weights'].iloc[permutation_order].values
    # data['rh_98'] = data['rh_98'].iloc[permutation_order].values
    # data['rh_99'] = data['rh_99'].iloc[permutation_order].values
    # data['rh_100'] = data['rh_100'].iloc[permutation_order].values
    # data['als_h'] = data['als_h'].iloc[permutation_order].values
    results.append(plot_data(data, name='data'))

print(">>>>>>>>>")
counter = 0
for result in results:
    if result.slope > 0.75:
        counter += 1
        print(result)
print("Length of significant ", counter)

plot_data(original_data, name='data')

# plot_data(cluster1, name='cluster1')
# plot_data(cluster2, name='cluster2')
# plot_data(cluster3, name='cluster3')
# plot_data(cluster4, name='cluster4')
# plot_data(cluster5, name='cluster5')
# for i, part in  enumerate(cluster5_parts):
#     plot_data(part, name='part' + str(i))
