import numpy as np
from typing import Self


# FIXXXXXXXXXXXXXXXXXXX THIS IS NOT ABSOLUTE
# Going to assume they mean absolute as well
def mean_error(actual: np.array, predicted: np.array):
    return np.mean(actual - predicted)


def maximum_absolute_error(actual: np.array, predicted: np.array):
    actual = actual.ravel()
    predicted = predicted.ravel()

    abs_error = np.abs(actual - predicted)
    arg_max = np.argmax(abs_error)
    # print("Actual: ", actual[arg_max])
    # print("Predicted: ", predicted[arg_max])

    return abs_error[arg_max], arg_max


def correlation(actual, predicted):
    actual_mean, predicted_mean = np.mean(actual), np.mean(predicted)
    numerator = np.sum((actual - actual_mean) * (predicted - predicted_mean))
    denominator = np.sqrt(np.sum((actual - actual_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2))
    return numerator / denominator

def mse(actual: np.array, predicted: np.array):
    actual = actual.ravel()
    predicted = predicted.ravel()
    return np.mean((actual - predicted) ** 2)

# Root mean squared error
def rmse(actual: np.array, predicted: np.array):
    actual = actual.ravel()
    predicted = predicted.ravel()
    return np.sqrt(np.mean((actual - predicted) ** 2))


# @jit(nopython=True)
def determination(actual: np.array, predicted: np.array):
    mean_actual = np.mean(actual)

    residual_sum = np.sum((predicted - actual) ** 2)
    total_sum = np.sum((actual - mean_actual) ** 2)

    return 1 - residual_sum / total_sum


class Error:
    def __init__(self, score, mae, root_err, mean_err):
        self.score = score
        self.mae = mae
        self.root_err = root_err
        self.mean_err = mean_err

    @staticmethod
    def calculate_error(y, y_predict) -> Self:
        score = determination(y, y_predict)

        mae = maximum_absolute_error(y, y_predict)
        root_err = rmse(y, y_predict)
        mean_err = mean_error(y, y_predict)
        return Error(score, mae[0], root_err, mean_err)

    def __str__(self):
        return f"Score: {self.score}, MAE: {self.mae}, RMSE: {self.root_err}, MeanE: {self.mean_err}"

    def __repr__(self):
        return self.__str__()
