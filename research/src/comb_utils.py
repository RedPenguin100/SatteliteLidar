from numba import njit

import numpy as np


@njit
def n_choose_r(n: int, r: int) -> int:
    if r > n:
        return 0
    result = 1
    for i in range(1, r + 1):
        result = result * (n - i + 1) // i
    return result


@njit
def combinations_numba(arr, r):
    """
    Generate all r-combinations of the elements in a 1D NumPy array 'arr'.
    Returns a 2D array where each row is one combination.
    """
    n = arr.shape[0]
    total = n_choose_r(n, r)
    out = np.empty((total, r), dtype=arr.dtype)
    indices = np.empty(r, dtype=np.int64)
    for i in range(r):
        indices[i] = i
    count = 0
    while True:
        for j in range(r):
            out[count, j] = arr[indices[j]]
        count += 1
        found = False
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                found = True
                break
        if not found:
            break
    return out


@njit
def compositions_numba(r, d):
    """
    Generate all d-tuples of nonnegative integers that sum to r.

    Parameters:
      r : int
          The target sum.
      d : int
          Number of parts in each tuple.

    Returns:
      compositions : 2D np.ndarray of shape (N, d)
          Each row is a d-tuple that sums to r.

    This is done by considering r stars and d-1 bars distributed over r+d-1 slots.
    """
    if d == 1:
        out = np.empty((1, 1), dtype=np.int64)
        out[0, 0] = r
        return out

    n_slots = r + d - 1
    # Create an array [0, 1, ..., n_slots-1]
    indices = np.empty(n_slots, dtype=np.int64)
    for i in range(n_slots):
        indices[i] = i
    # Get all ways to choose d-1 positions (bars) out of n_slots
    bars = combinations_numba(indices, d - 1)
    total = bars.shape[0]
    compositions = np.empty((total, d), dtype=np.int64)

    for i in range(total):
        # The first part is the number of stars before the first bar.
        compositions[i, 0] = bars[i, 0]
        # For intermediate parts, the count is the gap between successive bars minus 1.
        for j in range(1, d - 1):
            compositions[i, j] = bars[i, j] - bars[i, j - 1] - 1
        # The last part is the number of stars after the last bar.
        compositions[i, d - 1] = n_slots - bars[i, d - 2] - 1
    return compositions


@njit
def my_polynomial_features(arr: np.array, m: int) -> np.array:
    if m == 0:
        return np.ones((arr.shape[0], 1), dtype=np.float64)

    columns_size = arr.shape[1]
    compositions = []
    for i in range(1, m + 1):
        compositions.extend(compositions_numba(i, columns_size))

    result = np.ones((arr.shape[0], 1 + len(compositions)), dtype=np.float64)
    for r, row in enumerate(arr):

        for i, composition in enumerate(compositions):
            for j, pow in enumerate(composition):
                result[r, 1 + i] *= row[j] ** pow

    return result
