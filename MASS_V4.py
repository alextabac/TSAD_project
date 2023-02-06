## Author: Alexandru Paul Tabacaru ; alextabac@gmail.com
## implemented from reference below, from Matlab to Python
## This is a MIT license doc, for reserach and study purposes only.

## Referecne:
## MASS_V4 Implementation from below reference/source-codes
# Abdullah Mueen, Sheng Zhong, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and
# Eamonn Keogh (2022), The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance,
# URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

# The Matlab code for reference above was created by Sheng Zhong and Abdullah Mueen.
# The overall time complexity of the code is O(n log n).
# The code is free to use for research purposes.
# The code does not produce imaginary numbers due to numerical errors,
# k should greater than or equals to floor((3m+1) / 2).

import numpy as np
import scipy.stats as stats
from math import floor, sqrt
from scipy.fft import dct

class MASS_V4:
    def __init__(self, q_size, k_size=0):
        self.m = q_size
        self.k = k_size
        if k_size < q_size:
            self.k = floor((3 * q_size + 1.0) / 2.0)

    def zNorm(self, Q):
        return stats.zscore(Q)

    def dist_prof(self, T, Q):
        """
        Euclidean distance metric.
        :param T:
        :param Q:
        :param k: should greater than or equals to floor((3m+1)/2)
        :return:
        """
        n = len(T)
        m = len(Q)
        Q = self.zNorm(Q)
        dist = np.array([])
        k = self.k
        batch = self.get_batch_size(k, m)
        for j in range(0, n - m + 2, batch - m + 1):
            right = j + batch  # -1 for Matlab code due to last index is included in slice contrary to Python
            if right >= n:
                right = n
            dot_p = self.dct_dot_product(T[j:right], Q)
            sigmaT = self.movstd(T[j:right], m-1)  # in Matlab they use w=1 such that normalized N and not N-1
            # sigmaT[np.isnan(sigmaT)] = 1.0  # consider making NaN values to 1.0, if appear any
            v = m - np.divide(dot_p, sigmaT)  # sigmaT here is equivalent to sigmaT[m:end] in Matlab
            d = np.sqrt(2.0 * v)
            dist = np.concatenate((dist, d))
        return dist

    def movstd(self, A, window):
        aw = np.lib.stride_tricks.sliding_window_view(A, window)
        return np.std(aw, axis=-1)

    def get_batch_size(self, k, m):
        b = floor((2.0 * k - 2.0) / 3.0) - 1
        if b < m:
            b = m
        pad_len = b + floor((b - m + 1.0) / 2.0) + floor((m + 1.0) / 2.0)
        while pad_len < k:
            b += 1
            pad_len = b + floor((b - m + 1.0) / 2.0) + floor((m + 1.0) / 2.0)
        if pad_len > k:
            b -= 1
        return int(b)

    def dct_dot_product(self, x, y):
        n = len(x)
        m = len(y)
        x_pad, y_pad, si = self.dct_padding(x, y)
        N = len(x_pad)
        xc = dct(x_pad, type=2, norm="ortho")  # From SciPy Note: need orthogonalize=False , but doesn't exist
        yc = dct(y_pad, type=2, norm="ortho")
        dct_product = np.multiply(xc, yc)
        dct_product.resize(N + 1)
        dct_product[N] = 0
        # dct_product[0] *= sqrt(2)  # mark out versus matlab implementation by author
        dot_p = dct(dct_product, type=1, norm="ortho")
        # dot_p[0] *= 2  # mark out versus matlab implementation by author
        # dot_p = sqrt(2 * N) * dot_p[si: si + n - m + 1]
        dot_p = dot_p[si: si + n - m + 1]  # the sqrt(2N) is mark out versus matlab implementation by author
        return dot_p

    def dct_padding(self, x, y):
        n = len(x)
        m = len(y)
        p2 = floor((n - m + 1) / 2)
        p1 = p2 + floor((m + 1) / 2)
        p4 = n - m + p1 - p2
        x_pad = np.zeros(p1 + n)
        x_pad[p1:] = x
        y_pad = np.zeros(m + p2 + p4)
        y_pad[p2: p2 + m] = y
        start_index = p1 - p2
        return x_pad, y_pad, start_index
