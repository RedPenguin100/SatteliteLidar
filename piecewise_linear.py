import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PiecewiseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    TODO:
    """

    def __init__(self, c):
        self.c = c

    def fit(self, X, y):
        """
        """
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        left_side = X < self.c
        right_side = ~left_side

        if not np.any(left_side) or not np.any(right_side):
            # TODO: handle this
            raise ValueError("C either too large or too small, problem is now regular regression")

        X_left = X[left_side]
        X_right = X[right_side]

        A_left = np.column_stack([np.ones_like(X_left), X_left, np.zeros_like(X_left)])

        A_right = np.column_stack(
            [np.ones_like(X_right), X_right, X_right - self.c])

        A = np.vstack((A_left, A_right))

        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        self.beta0_, self.beta1_, self.beta2_ = params

        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        y_pred = np.where(X < self.c,
                          self.beta0_ + self.beta1_ * X,
                          self.beta0_ + self.beta1_ * X + self.beta2_ * (X - self.c))
        return y_pred


model = PiecewiseLinearRegressor(3)

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 7, 9])

model.fit(X, y)
print(f"Model params: {model.beta0_}, {model.beta1_}, {model.beta2_}")
print(f" c={model.c}")

res = model.predict(X)

print(res)
