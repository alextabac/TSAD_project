
from math import floor
import numpy as np
from datetime import datetime
from MASS_V4 import MASS_V4
from scipy import signal
from scipy.signal import find_peaks


class DAMP_topK:
    def __init__(self, enable_print=True):
        self.enable_output = enable_print

    def DAMP_k(self, ts, discords_num):
        s_time = datetime.now()
        autocor, lags = DAMP_topK.xcorr(ts)
        subseq_len = int(DAMP_topK.find_max_peak_index(autocor[3010:4001], lags[3010:4001]))
        half_seqlen = int(floor(0.5 * subseq_len))
        mass_v4 = MASS_V4(subseq_len)
        curr_index = 1001
        N = len(ts)
        left_MP = np.zeros(N)
        best_so_far = -np.Inf
        bool_vec = np.ones(N, dtype=bool)
        lookahead = DAMP_topK.next_pow2(2 * subseq_len)
        if self.enable_output:
            print(f"Auto subsequence length set to {subseq_len}.")
            print(f"Starting from index {curr_index}, with lookahead of {lookahead}")
        # Handle the prefix to get a relatively high best so far discord score
        # Prefix for top k
        if self.enable_output:
            print("Prefix handled starting.")
        cnt = int(curr_index/500) - 1
        for i in range(curr_index, curr_index + lookahead + 1):
            if self.enable_output and cnt < int(i/500):
                cnt += 1
                print(f"left_MP iteration {i} out of {(curr_index + 16 * subseq_len)}")
            # Use the brute force for the left Matrix Profile value
            if (i + subseq_len) > N:
                break
            left_MP[i] = min(mass_v4.dist_prof(ts[:i + 1], ts[i:i + subseq_len]))
        if self.enable_output:
            print("Prefix part 1 done.")
        left_MP_copy = left_MP.copy()
        for k in range(discords_num):
            if self.enable_output:
                print(f"Prefix iteration k discord {k}.")
            imax = np.argmax(left_MP_copy)
            best_so_far = left_MP_copy[imax]
            discord_start = max(1, imax - half_seqlen) - 1
            discord_end = max(half_seqlen + 1, imax + half_seqlen) + 1
            left_MP_copy[discord_start: discord_end] = -np.Inf
        if self.enable_output:
            print("Prefix has been handled.")

        # Remaining test data except for the prefix
        cnt = int(curr_index + 16 * subseq_len + 1/500) - 1
        for i in range(curr_index + 16 * subseq_len + 1, N - subseq_len):
            # Skip the current iteration if the corresponding boolean value is 0,
            # otherwise execute the current iteration
            if not bool_vec[i]:
                left_MP[i] = left_MP[i-1]-0.00001
                continue
            if self.enable_output and cnt < int(i/500):
                cnt += 1
                print(f"process iteration {i} out of {(N - subseq_len)}")
            # Use the brute force for the left Matrix Profile value
            if (i + subseq_len) > N:
                break
            # Initialization for classic DAMP
            # Approximate leftMP value for the current subsequence
            approximate_distance = np.Inf
            # x indicates how long a time series to look backwards
            x = DAMP_topK.next_pow2(8 * subseq_len)
            # flag indicates if it is the first iteration of DAMP
            flag = True
            # expansion_num indicates how many times the search has been
            # expanded backward
            expansion_num = 0
            query = ts[i:i + subseq_len]

            # Classic DAMP
            while approximate_distance >= best_so_far:
                # Case 1: Execute the algorithm on the time series segment
                if expansion_num * subseq_len + i - x < 0:
                    approximate_distance = min(mass_v4.dist_prof(ts[:i], query))
                    left_MP[i] = approximate_distance
                    if approximate_distance > best_so_far:
                        best_so_far = approximate_distance
                        left_MP_copy = left_MP.copy()
                        for k in range(discords_num):
                            imax = np.argmax(left_MP_copy)
                            best_so_far = left_MP_copy[imax]
                            discord_start = max(1, imax - half_seqlen) - 1
                            discord_end = max(half_seqlen + 1, imax + half_seqlen) + 1
                            left_MP_copy[discord_start: discord_end] = -np.Inf
                    break
                else:
                    if flag:
                        # Case 2: Execute the algorithm on the time series
                        flag = False
                        approximate_distance = min(mass_v4.dist_prof(ts[i-x+1:i+1], query))
                    else:
                        # Case 3: All other cases
                        x_start = int(i - x + 1 + (expansion_num * subseq_len))
                        x_end = int(i - (x / 2) + (expansion_num * subseq_len))
                        approximate_distance = min(mass_v4.dist_prof(ts[x_start:x_end+1], query))
                    if approximate_distance < best_so_far:
                        left_MP[i] = approximate_distance
                        break
                    else:
                        x = 2 * x
                        expansion_num = expansion_num + 1

            if lookahead > 0:
                # Perform forward MASS for pruning
                # The index at the beginning of the forward mass should be avoided in the exclusion zone
                start_of_mass = int(min(i + subseq_len, N - 1))
                end_of_mass = int(min(start_of_mass + lookahead, N))
                if (end_of_mass - start_of_mass) >= subseq_len:
                    distance_profile = mass_v4.dist_prof(ts[start_of_mass:end_of_mass], query)
                    dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]  # get the array in the tuple
                    ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass
                    bool_vec[ts_index_less_than_BSF] = False  # prune these indices

        # Get pruning rate
        pv = bool_vec[curr_index: N - subseq_len + 1]
        if len(pv) > 0:
            pr = (len(pv)-sum(pv))/(len(pv))
        else:
            pr = 0
        # Get top-k discord
        e_time = datetime.now()
        d_time = e_time - s_time
        if self.enable_output:
            print(f"DAMP_topK run time {d_time}")
            print(f"Pruning Rate: {pr}")
        scores = []
        positions = []
        left_MP_copy = left_MP.copy()
        for k in range(discords_num):
            loc = np.argmax(left_MP_copy)
            val = left_MP_copy[loc]
            if val == 0.0:
                if self.enable_output:
                    print(f"Only {k} discords were found.")
                break
            if self.enable_output:
                print(f"Predicted discord score/position (top {k}): {val}/{loc}")
            scores.append(val)
            positions.append(loc)
            discord_start = max(1, loc - half_seqlen) - 1
            discord_end = max(half_seqlen + 1, loc + half_seqlen) + 1
            left_MP_copy[discord_start: discord_end] = -np.Inf
        return scores, positions, left_MP

    @staticmethod
    def xcorr(ts, max_lag=3000):
                corr = signal.correlate(ts, ts, mode="full")
                lags = signal.correlation_lags(max_lag, max_lag, mode="full")
                return corr, lags

    @staticmethod
    def find_max_peak_index(autocor, lags):
        idx = find_peaks(autocor)[0]
        if len(idx) > 0:
            new_corr = np.take(autocor, idx)
            i = np.argmax(new_corr)
            j = idx[i]
            return lags[j]
        else:
            return 1000

    @staticmethod
    def next_pow2(x):
        # 1 if x == 0 else 2 ** (x - 1).bit_length()  # but no need to worry about x==0
        return 2 ** (x - 1).bit_length()