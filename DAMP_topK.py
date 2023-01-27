
import numpy as np
from datetime import datetime
from MASS_V4 import MASS_V4
from scipy import signal
from scipy.signal import find_peaks

class DAMP_topK:
    def __init__(self, var_arg_in):
        self.varargin = dict(var_arg_in)
        self.lookahead = self.varargin["lookahead"]
        self.enable_output = self.varargin["enable_output"]
        self.start_loc = 0
        self.subseq_len = 0

    def DAMP_k(self, ts, k_discords, start_loc):
        s_time = datetime.now()
        self.start_loc = start_loc
        # self.subseq_len = subseq_len
        # self.initial_checks(T, subseq_len, start_loc)
        autocor, lags = self.xcorr(ts)
        self.subseq_len = self.find_max_peak_index(autocor[3010:4001], lags[3010:4001])
        curr_index = 1001

    def xcorr(self, ts, max_lag=3000):
        corr = signal.correlate(ts, ts, mode="full")
        lags = signal.correlation_lags(max_lag, max_lag, mode="full")
        return corr, lags

    def find_max_peak_index(self, autocor, lags):
        idx = find_peaks(autocor)[0]
        if len(idx) > 0:
            new_corr = np.take(autocor, idx)
            i = np.argmax(new_corr)
            j = idx[i]
            return lags[j]
        else:
            return 1000
