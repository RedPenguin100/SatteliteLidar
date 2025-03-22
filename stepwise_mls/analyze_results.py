import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    experiments_halton = pd.read_csv('experiment-hetero_experiment.csv')
    sorted_halton = experiments_halton.sort_values(by=['base_function', 'n', 'm', 'delta'])
    # err_columns = ['mls_i', 'mls_plus', 'mls_union']

    sine_experiments = experiments_halton[experiments_halton['base_function'] == 'sine']
    exp_experiments = experiments_halton[experiments_halton['base_function'] == 'exp']

    colors_reds = ['red', 'tomato', 'crimson', 'firebrick']
    colors_blues = ['blue', 'dodgerblue', 'cornflowerblue', 'royalblue']
    colors_greens = ['green', 'limegreen', 'seagreen', 'forestgreen']

    error = 'score'

    for m in [0,1,2]:
        for i, n in enumerate([50, 55, 60, 65]):
            crop = sine_experiments[(sine_experiments['n'] == n) & (sine_experiments['m'] == m)]

            # plt.plot(crop['delta'], crop[f'mls_i_{error}'], label='MLS on I', color=colors_reds[i])
            plt.plot(crop['delta'], crop[f'mls_plus_{error}'], label='MLS novel', color=colors_blues[i])
            plt.plot(crop['delta'], crop[f'mls_union_{error}'], label='MLS on IuJ', color=colors_greens[i])

        plt.xlabel('Delta values')
        plt.ylabel(f'Error {error}')
        plt.title(f'{error} errors w.r.t Delta. m={m}, func=sine\n (Multiple experiments w/ different sample sizes)')
        # plt.legend(['MLS on I', 'MLS novel', 'MLS on IuJ'])
        plt.legend(['MLS novel', 'MLS on IuJ'])

        plt.show()

    # print(sorted_halton)
    # print(sorted_halton_fast)
    # print(sorted_halton)
    # print(sorted_halton_fast)
    # errors = (sorted_halton_fast.reset_index(drop=True)[err_columns] - sorted_halton.reset_index(drop=True)[err_columns]) / sorted_halton_fast.reset_index(drop=True)[err_columns]
    # print(errors.sort_values(by=['mls_i']))

    # print(sorted_halton)

    # print(sorted_halton_fast[err_columns])
