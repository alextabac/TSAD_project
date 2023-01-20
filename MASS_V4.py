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
            print(f"max dot_p={np.max(dot_p)} ; min dot_p={np.min(dot_p)}")
            if sum(np.isnan(dot_p)) > 0:
                print(f"Found nan values from dct_dot_product: {sum(np.isnan(dot_p))} out of {len(dot_p)}")
            sigmaT = self.movstd(T[j:right], m)
            sigmaT[np.isnan(sigmaT)] = 1.0
            sigmaT[sigmaT < 1.0] = 1.0
            if sum(np.isnan(sigmaT)) > 0:
                print(f"Found nan values from movstd: {sum(np.isnan(sigmaT))} out of {len(sigmaT)}")
            ndiv = np.divide(dot_p, sigmaT)
            neg = (ndiv>m).sum()
            if neg > 0:
                print(f"found negative values in sqrt, total {neg}; m={m}")
            d = np.sqrt(2.0 * (m - np.divide(dot_p, sigmaT)))  # sigmaT[m:end] in Matlab
            if sum(np.isnan(d)) > 0:
                print(f"Found nan values from np.div: {sum(np.isnan(d))} out of {len(d)}")
            break
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
        if sum(np.isnan(x_pad)) > 0:
            print(f"Found nan values in x_pad: {sum(np.isnan(x_pad))} out of {len(x_pad)}")
        if sum(np.isnan(y_pad)) > 0:
            print(f"Found nan values in y_pad: {sum(np.isnan(y_pad))} out of {len(y_pad)}")
        print(f"len(x_pad)={len(x_pad)} ; len(y_pad)={len(y_pad)} ; si={si}")
        N = len(x_pad)
        xc = dct(x_pad, type=2)
        yc = dct(y_pad, type=2)
        if sum(np.isnan(xc)) > 0:
            print(f"Found nan values in xc: {sum(np.isnan(xc))} out of {len(xc)}")
        if sum(np.isnan(yc)) > 0:
            print(f"Found nan values in y_pad: {sum(np.isnan(yc))} out of {len(yc)}")
        dct_product = np.multiply(xc, yc)
        # print(f"len(xc)={len(xc)} ; len(yc)={len(yc)} ; len(dct_product)={len(dct_product)} ; N={N}")
        dct_product.resize(N + 1)
        dct_product[N] = 0
        dct_product[0] *= sqrt(2)
        dot_p = dct(dct_product, type=1)
        if sum(np.isnan(dot_p)) > 0:
            print(f"Found nan values in dot_p: {sum(np.isnan(dot_p))} out of {len(dot_p)}")
        # print(f"len(dot_p)={len(dot_p)}")
        dot_p[0] *= 2
        dot_p = sqrt(2 * N) * dot_p[si: si + n - m + 1]
        return dot_p

    def dct_padding(self, x, y):
        n = len(x)
        m = len(y)
        p2 = floor((n - m + 1) / 2)
        p1 = p2 + floor((m + 1) / 2)
        p4 = n - m + p1 - p2
        # print(f"p1={p1} ; p2={p2} ; p4={p4}")
        x_pad = np.zeros(p1 + n)
        x_pad[p1:] = x
        y_pad = np.zeros(m + p2 + p4)
        y_pad[p2: p2 + m] = y
        start_index = p1 - p2
        return x_pad, y_pad, start_index
