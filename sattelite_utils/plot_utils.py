import matplotlib.pyplot as plt


def plot_piecewise(result):
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
    plt.plot(x_left, y_left, color='red', linestyle='--', label=f'y = {result.slope:.2f}x + {result.intercept:.2f}')
    plt.plot(x_right, y_right, color='red', linestyle='--',
             label=f'y = {result.slope:.2f}x + {result.intercept:.2f}')

    plt.legend(loc='upper left')
    plt.show()


def plot_line(result, color_column='None'):
    if color_column == 'weights':
        plt.scatter(result.data['als_h'], result.data['rh_98'], c=result.weights, s=10)
        plt.colorbar(label=color_column)
    elif not color_column is None:
        plt.scatter(result.data['als_h'], result.data['rh_98'], c=result.data[color_column], s=10)
        plt.colorbar(label=color_column)
    else:
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
