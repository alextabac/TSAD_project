
#  the function xcorr2 was copied from https://github.com/colizoli/xcorr_python/blob/master/xcorr.py

import numpy as np
import pandas as pd
from scipy import signal


def get_clear_indices(idx, replace_indices, word_size, limit=np.Inf):
    if len(replace_indices) == 0:
        if limit < np.Inf:
            fl = [i for i in idx if i < limit]
            return fl
        else:
            return idx
    replace_indices.sort()
    ss = [(max(0, i - word_size), 1) for i in replace_indices]  # 1 == start
    se = [(i + word_size, 2) for i in replace_indices]  # 2 == end
    sl = ss + se
    sl.sort(key=lambda x: x[0])
    ll = []
    o = 0
    s = sl[0][0]
    for e in sl:
        if e[1] == 1:
            o += 1
            if o == 1:
                s = e[0]
        elif e[1] == 2:
            o -= 1
        if o == 0:
            e = e[0]
            ll.append((s, e))
    fl = []
    j = 0
    n = len(ll) - 1
    for i in idx:
        while j < n and i > ll[j][1]:
            j += 1
        if i < ll[j][0] or i >= ll[j][1]:
            if i < limit:
                fl.append(i)
    return fl

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


def get_hotsax_appearances_matrix(hotsax_obj):
    rows_l = []
    for a in range(3, 11):
        for w in range(3, 33):
            hotsax_obj.set_SAX_params(a, w)
            hotsax_obj.init_norm()
            d = {'alpha': a, 'word': w}
            for i in range(1, 33):
                i_s = 'appearing_' + str(i) + '_times'
                d[i_s] = len(np.where(np.array(hotsax_obj.sax_array['count'].values) == i)[0])
            rows_l.append(d)
    return pd.DataFrame(rows_l)
