import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_hdf('data/data.h5')
# data = data[data['Longitude'] >= -54.]


X = data['rh_98'].to_numpy().reshape(-1, 1)
Y = data['als_h'].to_numpy()

model = LinearRegression()
model.fit(X, Y)


# def filter_data(data):
#     return data[data['Longitude'] >= -54.]

def filter_data(data):
    return data[(data['Longitude'] <= -54.7) & (data['Longitude'] >= -54.88)]


def filter_beam(data, beam):
    return data[data['BEAM'] == beam]


data_filtered = filter_data(data)
# data_filtered = data


def plot_data(data, model):
    X_spot = data['rh_98'].to_numpy().reshape(-1, 1)
    Y_pred = model.predict(X_spot)

    Y = data['als_h'].to_numpy()

    print("Model score: ", model.score(X_spot, Y))

    errors = np.abs(Y - Y_pred)

    longitudes = data['Longitude']
    latitudes = data['Latitude']

    plt.figure(figsize=(8, 6))
    sizes = 10 + Y * 20
    plt.scatter(longitudes, latitudes, c=errors, cmap='gray', s=sizes, edgecolor='black', vmin=0, vmax=20)

    # Add vertical lines above each point with lengths proportional to `errors`
    plt.vlines(longitudes, latitudes, latitudes + (data['sensitivity'] ** 30) / 20., color='red', lw=2)
    # plt.vlines(longitudes + np.ones_like(longitudes) * 0.01, latitudes, latitudes + (data['sensitivity'] ** 30) / 20., color='red', lw=2)

    plt.colorbar(label='Error Magnitude')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample with errors')
    plt.show()


print("Total data")
plot_data(data, model)
print("Filtered data")
plot_data(data_filtered, model)

X_filtered = data_filtered['rh_98'].to_numpy().reshape(-1, 1)
Y_filtered = data_filtered['als_h'].to_numpy()
model_filtered = LinearRegression()
model_filtered.fit(X_filtered, Y_filtered)

print("Filtered model")
plot_data(data_filtered, model_filtered)
print("Filtered model total data")
plot_data(data, model_filtered)

for beam in range(7):
    data_beam = filter_beam(data_filtered, beam)
    print(f"Beam #{beam}")
    if len(data_beam) == 0:
        print("Skipping beam")
        continue
    plot_data(data_beam, model_filtered)

    model_beam = LinearRegression()
    model_beam.fit(data_beam['rh_98'].to_numpy().reshape(-1, 1), data_beam['als_h'].to_numpy())
    print(f"Beam filtered #{beam}")
    plot_data(data_beam, model_beam)

    print(f"Beam with original model #{beam}")
    plot_data(data_beam, model)

print("Als-h mean: ", np.mean(data['als_h']))
print("Als-h std: ", np.std(data['als_h']))
print("rh98 mean: ", np.mean(data['rh_98']))
print("rh98 std: ", np.std(data['rh_98']))
print("model mean: ", np.mean(model.predict(data['rh_98'].to_numpy().reshape(-1, 1))))
print("model std: ", np.std(model.predict(data['rh_98'].to_numpy().reshape(-1, 1))))
