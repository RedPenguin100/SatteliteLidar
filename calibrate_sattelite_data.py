import math

import pandas as pd
import numpy as np
from numba import jit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from piecewise_linear import PiecewiseLinearRegressor
from timer import Timer
from sklearn.linear_model import LinearRegression, QuantileRegressor
import matplotlib.pyplot as plt
from matplotlib.table import Table
from error_utils import *


class Result:
    def __init__(self, slope, intercept, err: Error, exp_name='', data=None, weights=None, params=None):
        self.slope = slope
        self.intercept = intercept
        self.err: Error = err
        self.exp_name = exp_name
        self.data = data
        self.weights = weights
        self.params = params

    def __str__(self):
        if self.exp_name == '':
            return f"Slope={self.slope}, intercept={self.intercept}, Err: [{self.err}]"
        return f"Experiment={self.exp_name}, slope={self.slope}, intercept={self.intercept}, Error: [{self.err}]"

    def __repr__(self):
        return self.__str__()


def piecewise_linear_squares(x: np.array, y: np.array, exp_name='', data=None):
    x = np.asarray(x).ravel()
    param_grid = {
        'c': x[(x > 1) & (x < 15)]
    }

    # model = PiecewiseLinearRegressor(c=None)
    #
    #
    # grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    # grid_search.fit(x, y)
    # best_model = grid_search.best_estimator_

    best_model = None
    max_score = np.inf
    #     for x_point in x[(x > 1)  & (x < 15)]:
    for x_point in np.sort(x)[1:-1]:
        model = PiecewiseLinearRegressor(c=x_point)
        model.fit(x, y)
        y_predict = model.predict(x)
        score = rmse(y, y_predict)
        if score < max_score:
            max_score = score
            best_model = model

    best_model.beta0_ = 2.5
    best_model.beta1_ = 0.6
    best_model.beta2_ = 0.4
    best_model.c = 8
    y_predict = best_model.predict(x)

    print(f"Params: ", [best_model.beta0_, best_model.beta1_, best_model.beta2_, best_model.c])

    return Result(-1, -1, Error.calculate_error(y, y_predict), exp_name=exp_name, data=data,
                  params=[best_model.beta0_, best_model.beta1_, best_model.beta2_, best_model.c])


def least_squares(x: np.array, y: np.array, exp_name='', data=None):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_predict = model.predict(x)

    return Result(model.coef_[0], model.intercept_, Error.calculate_error(y, y_predict),
                  exp_name=exp_name, data=data, params=[model.coef_[0], model.intercept_])


def weighted_least_squares(x: np.array, w: np.array, y: np.array, exp_name='', data=None) -> Result:
    x = x.reshape(-1, 1)
    sqrt_weights = np.sqrt(w)
    x_tag = x * sqrt_weights[:, np.newaxis]
    y_tag = y * sqrt_weights

    model = LinearRegression()
    model.fit(x_tag, y_tag)

    # y_predict = model.predict(x_tag) / sqrt_weights
    y_predict = model.predict(x)
    # print("Exp name: ", exp_name)
    # print("Mean: ", np.mean(y_predict))
    # print("std: ", np.std(y_predict))

    return Result(model.coef_[0], model.intercept_, Error.calculate_error(y, y_predict),
                  exp_name=exp_name, data=data, weights=w, params=[model.coef_[0], model.intercept_])


def regression_l1(x: np.array, y: np.array, exp_name='', sensitivity=True, geo_sensitivity=False, quality=False,
                  data=None):
    x = x.reshape(-1, 1)

    final_weights = np.ones_like(y)

    if sensitivity:
        sensitivity_vec = data['sensitivity'].to_numpy()
        sensitivity_weights = sensitivity_vec ** 2
        final_weights = final_weights * sensitivity_weights

    if geo_sensitivity:
        geo_sensitivity_vec = data['geolocation_sensitivity_a2'].to_numpy()
        geo_sensitivity_weights = geo_sensitivity_vec ** 2
        final_weights = final_weights * geo_sensitivity_weights

    if quality:
        quality_weights = data['quality_flag'].to_numpy()
        final_weights = quality_weights * final_weights

    x = x[final_weights != 0]
    y = y[final_weights != 0]
    final_weights = np.asarray(final_weights[final_weights != 0])

    model = QuantileRegressor(quantile=0.5, alpha=0.)
    model.fit(x, y, sample_weight=final_weights)

    y_predict = model.predict(x)

    return Result(model.coef_[0], model.intercept_, Error.calculate_error(y, y_predict),
                  exp_name=exp_name, data=data, params=[model.coef_[0], model.intercept_])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_variance_weights(data: pd.DataFrame):
    rh_98 = data['rh_98'].to_numpy()
    rh_99 = data['rh_99'].to_numpy()
    rh_100 = data['rh_100'].to_numpy()
    stacked = np.vstack([rh_98, rh_99, rh_100])
    stacked = stacked / np.mean(stacked, axis=0)
    variance_vector = np.var(stacked, axis=0)

    # variance_weights = (sigmoid(1 / variance_vector) * 2)
    # variance_weights = np.sqrt(np.clip(1 / variance_vector, a_min=None, a_max=1))

    variance_max = (np.max(variance_vector) - np.min(variance_vector))
    # # print("Variance_distance", variance_max)
    # # print("Variance_min", np.min(variance_vector))
    # # print("variance_max", np.max(variance_vector))
    variance_weights = 1 - 0.9 * (variance_vector - np.min(variance_vector)) / variance_max
    # variance_weights *= variance_weights
    return variance_weights


def sigmoid_importance(x, a=60, b=0.93):
    return 1 / (1 + np.exp(-a * (x - b)))


def calculate_weights(data: pd.DataFrame, quality_capped=False, site_capped=False, sensitivity_geo=False,
                      small_tree=False,
                      quality=False, sensitivity=True, site=False):
    variance_weights = get_variance_weights(data)

    final_weights = variance_weights

    if sensitivity:
        sensitivity = data['sensitivity'].to_numpy()
        sensitivity_weights = sensitivity ** 4
        final_weights = final_weights * sensitivity_weights

    if sensitivity_geo:
        sensitivity_geo = data['geolocation_sensitivity_a2'].to_numpy()
        # sensitivity_geo_weights = sensitivity_geo ** 4
        sensitivity_geo_weights = sigmoid_importance(sensitivity_geo)
        final_weights = final_weights * sensitivity_geo_weights

    if quality:
        quality_weights = data['quality_flag'].to_numpy()
        final_weights = quality_weights * final_weights

    if quality_capped:
        quality_weights = data['quality_flag'].to_numpy()
        final_weights = np.clip(quality_weights, a_min=0.5, a_max=1) * final_weights

    if site_capped:
        site_weights = data['Site'].to_numpy()
        final_weights = np.clip(site_weights, a_min=0.8, a_max=1) * final_weights

    if site:
        site_weights = data['Site'].to_numpy()
        final_weights = final_weights * site_weights

    if small_tree:
        small_tree_weights = (data['als_h'] > 1.).astype(np.float64)
        final_weights = final_weights * small_tree_weights

    return final_weights


def fit_data(data: pd.DataFrame, quality_capped=False, site_capped=False, sensitivity_geo=False, small_tree=True,
             quality=False,
             sensitivity=True, site=False, exp_name='') -> Result:
    als_h = data['als_h'].to_numpy()
    rh_98 = data['rh_98'].to_numpy()
    rh_99 = data['rh_99'].to_numpy()
    rh_100 = data['rh_100'].to_numpy()

    if 'weights' not in data.columns:
        final_weights = calculate_weights(data, quality_capped=quality_capped, site_capped=site_capped,
                                          sensitivity=sensitivity,
                                          sensitivity_geo=sensitivity_geo, small_tree=small_tree, quality=quality,
                                          site=site)
    else:
        final_weights = data['weights']

    # rh_100 = rh_100[final_weights != 0]
    rh_98 = rh_98[final_weights != 0]
    als_h = als_h[final_weights != 0]
    data_filtered = data[final_weights != 0]
    final_weights = np.asarray(final_weights[final_weights != 0])
    if len(final_weights) == 0:
        raise ValueError("Can't fit")
    return weighted_least_squares(x=als_h, w=final_weights, y=rh_98, exp_name=exp_name, data=data_filtered)


def fit_data_least(data: pd.DataFrame, exp_name=''):
    als_h = data['als_h'].to_numpy()
    rh_98 = data['rh_98'].to_numpy()
    return least_squares(x=als_h, y=rh_98, exp_name=exp_name, data=data)


if __name__ == '__main__':

    data = pd.read_hdf('./data/data.h5')

    print(data.columns)

    als_h = data['als_h'].to_numpy()  # plane data
    rh_98 = data['rh_98'].to_numpy()
    rh_99 = data['rh_99'].to_numpy()
    rh_100 = data['rh_100'].to_numpy()
    rh_list = [(rh_98, 'rh98'), (rh_99, 'rh99'), (rh_100, 'rh100')]
    sensitivity = data['sensitivity'].to_numpy()

    print("Data length", len(data))

    results_list = []

    for rh, rh_name in rh_list:
        result = least_squares(x=als_h, y=rh, exp_name=rh_name, data=data)
        print(f"Least squares {rh_name} ", result)
        results_list.append(result)

    results_list.append(
        regression_l1(data['als_h'].to_numpy(), data['rh_98'].to_numpy(), sensitivity=False, exp_name='Michael-L1-Base',
                      data=data))
    results_list.append(regression_l1(data['als_h'].to_numpy(), data['rh_98'].to_numpy(), sensitivity=True,
                                      exp_name='Michael-L1-Sensitivity', data=data))

    results_list.append(
        piecewise_linear_squares(data['als_h'].to_numpy(), data['rh_98'].to_numpy(), exp_name='Michael-Piecewise',
                                 data=data))

    results_list.append(
        regression_l1(data['als_h'].to_numpy(), data['rh_98'].to_numpy(), sensitivity=True, geo_sensitivity=True,
                      exp_name='Michael-L1-Geo+Sensitivity', data=data))

    results_list.append(
        regression_l1(data['als_h'].to_numpy(), data['rh_98'].to_numpy(), sensitivity=True, geo_sensitivity=True,
                      quality=True,
                      exp_name='Michael-L1-Best', data=data))

    data['var'] = get_variance_weights(data)

    # site_quality_capped = {'quality_capped' : True, 'site_capped' : True}
    kwargs = {'sensitivity': False, 'sensitivity_geo': True, 'site': False, 'quality': True}
    # kwargs = {**kwargs, **site_quality_capped}

    results_list.append(
        fit_data(data, small_tree=False, quality=False, site=False, sensitivity=False, exp_name='Michael-Var-Base'))

    # data['weights'] = 0
    var_best = fit_data(data, small_tree=False,
                        exp_name='Michael-Best', **kwargs)

    data_filtered = var_best.data
    data_filtered['weights'] = var_best.weights
    # data[data['quality_flag'] == 1]['weights'] = var_best.weights

    results_list.append(fit_data(data, small_tree=False, quality=False, site=False, exp_name='Michael-Var-Sensitivity'))

    results_list.append(
        fit_data(data, sensitivity_geo=True, sensitivity=False, small_tree=False, quality=False, site=False,
                 exp_name='Michael-Var-Geo'))

    results_list.append(fit_data(data, sensitivity_geo=True, small_tree=False, quality=False, site=False,
                                 exp_name='Michael-Var-Sensitivity+Geo'))

    results_list.append(var_best)

    results_list.append(
        fit_data(data_filtered.sort_values(by='weights').tail(500), small_tree=False,
                 exp_name='Michael-Weights-Head500',
                 **kwargs))

    results_list.append(
        fit_data(data_filtered.sort_values(by='weights').tail(300), small_tree=False,
                 exp_name='Michael-Weights-Head300',
                 **kwargs))

    results_list.append(
        fit_data(data_filtered.sort_values(by='weights').tail(150), small_tree=False,
                 exp_name='Michael-Weights-Head150',
                 **kwargs))

    results_list.append(
        fit_data(data, small_tree=False, quality=True, site=False, exp_name='Michael-Var-QUALITY-FILTER'))
    results_list.append(fit_data(data, small_tree=False, quality=False, site=True, exp_name='Michael-Var-SITE-FILTER'))
    results_list.append(
        fit_data(data, small_tree=False, quality=True, site=True, exp_name='Michael-Var-QUALITY-SITE-FILTER'))

    data_quality = data[data['quality_flag'] == 1]
    print("High quality: ", len(data_quality))

    results = fit_data_least(data_quality)
    print("Quality data results old", results)

    results = fit_data(data_quality)
    print("Quality data results new", results)

    data_quality = data[data['quality_flag'] == 0]
    print("Low quality: ", len(data_quality))
    results = fit_data(data_quality, quality=False)
    print("Low Quality data results ", results)

    data['diff'] = np.abs(data['rh_98'] - data['als_h'])

    data['var'] = get_variance_weights(data)
    X = data.drop(columns=['diff', 'als_h', 'rh_98', 'rh_99', 'rh_100'])
    y = data['diff']
    model = RandomForestRegressor()
    model.fit(X, y)

    df = pd.DataFrame({'names': X.columns, 'values': model.feature_importances_})
    # from sklearn.inspection import PartialDependenceDisplay, partial_dependence
    #
    # deciles = {0: np.linspace(0, 1, num=10)}
    # pd_results = partial_dependence(model, X, features=0, kind='average', grid_resolution=10)
    # display = PartialDependenceDisplay([pd_results], features=model.feature_importances_, feature_names=df['names'], target_idx=0, deciles=deciles)
    # display.plot(pdp_lim={'var': (-1., 1.)})
    print(df.sort_values(by='values', ascending=False))

    X = data.drop(columns=['als_h'])
    y = fit_data(data, small_tree=False, quality=False, site=False, exp_name='Michael-ALL').data
    model = RandomForestRegressor()
    model.fit(X, y)

    print(X.columns)
    print(model.feature_importances_)
    print(sum(model.feature_importances_))

    # Sample DataFrame with similar structure as shown in the image
    data = {
        'Subgroup Name': ['ALL', 'QS90', 'QS95', 'QS98', 'QS99', 'ALL-S2', 'QS90-S2', 'QS95-S2', 'QS98-S2', 'QS99-S2'],
        '# Shots': [841, 496, 443, 240, 88, 841, 496, 478, 298, 112],
        '# Orbits': [36, 22, 22, 20, 16, 36, 22, 22, 20, 18],
        'MAE [m]': [49.44, 30.99, 21.87, 21.12, 21.12, 49.44, 30.99, 21.87, 21.12, 21.12],
        'H_Ref [m]': [24.14, 23.94, 24.47, 24.90, 25.58, 24.14, 23.94, 23.91, 24.64, 24.68],
        'MeanE [m]': [-2.62, -1.14, -0.91, -0.07, 0.53, -2.62, -1.14, -0.85, 0.04, 0.66],
        'RMSE [m]': [7.93, 5.40, 4.91, 4.25, 3.83, 7.93, 5.40, 4.97, 4.14, 3.61],
        'Slope [-]': [0.65, 0.74, 0.75, 0.74, 0.73, 0.65, 0.74, 0.76, 0.80, 0.80],
        'Int. [m]': [5.85, 5.07, 5.25, 6.47, 7.55, 5.85, 5.07, 4.77, 5.08, 5.60],
        'R^2': [0.38, 0.61, 0.61, 0.67, 0.68, 0.38, 0.61, 0.66, 0.74, 0.77]
    }

    for result in results_list:
        data['R^2'].append(round(result.err.score, 2))
        data['MAE [m]'].append(round(result.err.mae, 2))
        data['Int. [m]'].append(round(result.intercept, 2))
        data['Slope [-]'].append(round(result.slope, 2))
        data['RMSE [m]'].append(round(result.err.root_err, 2))
        data['MeanE [m]'].append(round(result.err.mean_err, 2))
        data['H_Ref [m]'].append(round(np.average(result.data['als_h']), 2))
        data['# Orbits'].append(math.nan)
        data['# Shots'].append(len(result.data['als_h']))  # TODO
        data['Subgroup Name'].append(result.exp_name)

    for result in results_list:
        if 'Head' in result.exp_name or 'Michael-Var-Sensitivity+Geo' in result.exp_name or 'Best' in result.exp_name:
            plt.scatter(result.data['als_h'], result.data['rh_98'], s=10)
            plt.xlabel('als_h')
            plt.ylabel('rh_98')
            plt.axis('scaled')

            max_value = 55
            min_value = 0
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            if result.exp_name == 'Michael-Var-Sensitivity+Geo':
                plt.title("All points(842)")
            elif result.exp_name == 'Michael-Best':
                plt.title("Only quality(753)")
            else:
                plt.title(result.exp_name)

            x_vals = [min_value, max_value]
            y_vals = [result.slope * x + result.intercept for x in x_vals]
            plt.plot(x_vals, y_vals, color='red', linestyle='--',
                     label=f'y = {result.slope:.2f}x + {result.intercept:.2f}')

            plt.legend(loc='upper left')
            plt.show()
        if "Piecewise" in result.exp_name:
            beta0, beta1, beta2, c = result.params

            plt.scatter(result.data['als_h'], result.data['rh_98'], s=10)
            plt.xlabel('als_h')
            plt.ylabel('rh_98')
            plt.axis('scaled')

            max_value = 55
            min_value = 0
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            if result.exp_name == 'Michael-Var-Sensitivity+Geo':
                plt.title("All points(842)")
            elif result.exp_name == 'Michael-Best':
                plt.title("Only quality(753)")
            else:
                plt.title(result.exp_name)

            x_left = [min_value, c]
            x_right = [c, max_value]
            y_left = [beta0 + beta1 * x for x in x_left]
            y_right = [beta0 + beta1 * x + beta2 * (x - c) for x in x_right]
            plt.plot(x_left, y_left, color='red', linestyle='--',
                     label=f'y = {result.slope:.2f}x + {result.intercept:.2f}')
            plt.plot(x_right, y_right, color='red', linestyle='--',
                     label=f'y = {result.slope:.2f}x + {result.intercept:.2f}')

            plt.legend(loc='upper left')
            plt.show()

    df = pd.DataFrame(data)


    def plot_data_as_table(df):
        # Plot setup with a larger figure size
        fig, ax = plt.subplots(figsize=(10, 6))  # Large figure size
        ax.axis('off')  # Turn off the axes

        # Create the table with a large font size
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)  # Disable automatic font size
        table.set_fontsize(10)  # Set a large font size for readability
        table.scale(1, 1)  # Scale the table to make cells larger
        # Make the first column wider
        for i in range(len(df) + 1):  # +1 to include the header row
            table[i, 0].set_width(0.3)  # Set width of first column cells
        for j in range(1, len(df.columns)):
            for i in range(len(df) + 1):
                table[i, j].set_width(0.1)  # Set width for other columns

        plt.show()


    plot_data_as_table(df)
