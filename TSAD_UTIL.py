
#  the function xcorr2 was copied from https://github.com/colizoli/xcorr_python/blob/master/xcorr.py

import numpy as np
from scipy import signal


def contains_constant_regions(ts, subsequence_len):
    # Assuming T has reset index
    bool_vec = False
    constant_indices = np.ediff1d(np.where(np.ediff1d(ts) == 0))
    serial_indices = np.split(constant_indices, np.where(constant_indices > 1)[0])
    lengths = [len(arr) for arr in serial_indices]
    max_len = max(lengths)
    if max_len >= subsequence_len or np.var(ts) < 0.2:
        bool_vec = True
    return bool_vec


def xcorr(ts, max_lag=3000):
    corr = signal.correlate(ts, ts, mode="full")
    lags = signal.correlation_lags(max_lag, max_lag, mode="full")
    return corr, lags


def xcorr2(ts, max_lag=3000):
    nx = len(ts)
    corr = np.correlate(ts, ts, mode='full')
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr[nx - 1 - max_lag: nx + max_lag]
    return corr, lags


def find_max_peak_index(autocor, lags):
    idx = signal.find_peaks(autocor)[0]
    if len(idx) > 0:
        new_corr = np.take(autocor, idx)
        i = np.argmax(new_corr)
        j = idx[i]
        return lags[j]
    else:
        return 1000


def next_pow2(x):
    # 1 if x == 0 else 2 ** (x - 1).bit_length()  # but no need to worry about x==0
    return 2 ** (x - 1).bit_length()

