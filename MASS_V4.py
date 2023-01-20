import numpy as np
import scipy.stats as stats
from math import floor, sqrt
from scipy.fft import dct


class MASS_V4:

    def zNorm(self, Q):
        return stats.zscore(Q)

    def get_similarities(self, T, Q, k):
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
        batch = self.get_batch_size(k, m)
        for j in range(0, n - m + 1, batch - m + 1):
            right = j + batch  # -1 for Matlab code due to last index is included in slice contrary to Python
            if right >= n:
                right = n
            dot_p = self.dct_dot_product(T[j:right], Q)
            sigmaT = self.movstd(T[j:right], m)  # in Matlab they use w=1 such that normalized N and not N-1
            # sigmaT[np.isnan(sigmaT)] = 1.0
            d = np.sqrt(2.0 * (m - np.divide(dot_p, sigmaT)))  # sigmaT[m:end] in Matlab
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
        dct_product[0] *= sqrt(2)
        dot_p = dct(dct_product, type=1, norm="ortho")
        dot_p[0] *= 2
        dot_p = sqrt(2 * N) * dot_p[si: si + n - m + 1]
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
