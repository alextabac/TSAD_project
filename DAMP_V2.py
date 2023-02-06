
import sys
from os import path
fpath = path.dirname(path.abspath(__file__))
if fpath not in sys.path:
    sys.path.append(fpath)
from datetime import datetime
from importlib import reload
import MASS_V4
reload(MASS_V4)
from MASS_V4 import MASS_V4
import TSAD_UTIL
reload(TSAD_UTIL)
from TSAD_UTIL import *


class DAMP_V2:
    def __init__(self, var_arg_in):
        self.varargin = dict(var_arg_in)
        self.lookahead = self.varargin["lookahead"]
        self.enable_output = self.varargin["enable_output"]
        self.start_loc = 0
        self.subseq_len = 0
        self.left_mp_result = None
        self.position_found = None
        self.score_found = None

    def progressive_damp(self, T, subseq_len, start_loc=2000, step_size=1000, verbose=False):
        scores = []
        poses = []
        windows = []
        dtimes = []
        end_index = len(T) - subseq_len + 1
        i = start_loc
        while i <= end_index:
            discord_score, position, left_MP, d_time = self.DAMP_2_0(T[:i], subseq_len, 1000)
            scores.append(discord_score)
            poses.append(position)
            windows.append(i)
            dtimes.append(d_time)
            if verbose:
                print(f"completed step {i} out of {end_index}; run-time {d_time}; discord: {position}")
            if i == end_index:
                break
            i += step_size
            i = min(i, end_index)
        return scores, poses, windows, dtimes

    def DAMP_2_0(self, T, subseq_len, start_loc):
        s_time = datetime.now()
        self.start_loc = start_loc
        self.subseq_len = subseq_len
        self.initial_checks(T, subseq_len, start_loc)  # lookahead; constant region; subsequence length;
        mass_v4 = MASS_V4(subseq_len)
        start_loc = self.start_loc
        subseq_len = self.subseq_len
        # self.lookahead = next_pow2(16 * subseq_len)
        lookahead = self.lookahead
        N = len(T)
        left_MP = np.zeros(N)
        best_so_far = -np.Inf
        bool_vec = np.ones(N, dtype=bool)
        # st1 = datetime.now()
        # print("starting treat for indices up to starting locaiton.")
        for i in range(start_loc, (start_loc + lookahead)):
            # Skip the current iteration if the corresponding boolean value
            if not bool_vec[i]:
                left_MP[i] = left_MP[i-1]-0.00001
                continue
            if (i + subseq_len) >= N:
                break
            query = T[i:(i+subseq_len)]
            left_MP[i] = min(mass_v4.dist_prof(T[:i+1], query))
            best_so_far = max(left_MP)
            # if lookahead is zero then it is pure online algorithm with no pruning
            if lookahead > 0:
                # perform forward MASS for pruning
                start_of_mass = min(i + subseq_len, N - 1)
                end_of_mass = min(start_of_mass + lookahead, N)
                if (end_of_mass - start_of_mass) > subseq_len:
                    distance_profile = mass_v4.dist_prof(T[start_of_mass:end_of_mass], query)
                    dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]  # get the array in tuple
                    ts_index_less_than_BSF_ = dp_index_less_than_BSF + start_of_mass
                    ts_index_less_than_BSF = [kk for kk in ts_index_less_than_BSF_ if kk < N]
                    bool_vec[ts_index_less_than_BSF] = False  # prune these indices
        # et1 = datetime.now()
        # print(f"Starting the actual search, so far delta time: {(et1-st1)}")
        # j = 0
        # st1 = datetime.now()
        for i in range(start_loc + lookahead, N - subseq_len):
            # j += 1
            # if j % 500 == 0:
            #     et1 = datetime.now()
            #     print(f"starting search iteration {i} , delta time: {et1-st1}; counted so far {j} cycles.")
            #     st1 = datetime.now()
            if not bool_vec[i]:
                # We subtract a very small number here to avoid the pruned
                # subsequence having the same discord score as the real discord
                left_MP[i] = left_MP[i-1] - 0.00001
                continue
            approximate_distance = np.Inf
            X = next_pow2(4 * subseq_len)
            flag = True
            expansion_num = 0
            if i + subseq_len >= N:
                break
            query = T[i:i + subseq_len]

            # Classic DAMP
            while approximate_distance >= best_so_far:
                # Case 1: Execute the algorithm on the time series segment
                if expansion_num * subseq_len + i - X < 0:
                    approximate_distance = min(mass_v4.dist_prof(T[:i+1], query))
                    left_MP[i] = approximate_distance
                    if approximate_distance > best_so_far:
                        best_so_far = approximate_distance
                    break
                else:
                    if flag:
                        # Case 2: Execute the algorithm on the time series
                        flag = False
                        approximate_distance = min(mass_v4.dist_prof(T[i-X+1:i+1], query))
                    else:
                        # Case 3: All other cases
                        X_start = int(i - X + (expansion_num * subseq_len) + 1)
                        X_end = int(i - (X / 2) + (expansion_num * subseq_len) + 1)
                        approximate_distance = min(mass_v4.dist_prof(T[X_start:X_end], query))
                    if approximate_distance < best_so_far:
                        left_MP[i] = approximate_distance
                        break
                    else:
                        X = 2 * X
                        expansion_num += 1
            if lookahead > 0:
                # Perform forward MASS for pruning
                # The index at the beginning of the forward mass should be avoided in the exclusion zone
                start_of_mass = int(min(i + subseq_len, N - 1))
                end_of_mass = int(min(start_of_mass + lookahead, N))
                if (end_of_mass - start_of_mass) > subseq_len:
                    distance_profile = mass_v4.dist_prof(T[start_of_mass:end_of_mass], query)
                    dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]  # get the array in tuple
                    ts_index_less_than_BSF_ = dp_index_less_than_BSF + start_of_mass
                    ts_index_less_than_BSF = [kk for kk in ts_index_less_than_BSF_ if kk < N]
                    # if len(ts_index_less_than_BSF) > 0 and max(ts_index_less_than_BSF) >= N:
                    #     maxi = max(ts_index_less_than_BSF)
                    #     print(f"exceed: i={i}; subseq_len={subseq_len}; N={N}; start_of_mass={start_of_mass}; " +
                    #           f"end_of_mass={end_of_mass}; len(dp)={len(distance_profile)}")
                    #     print(f"max index: {maxi}; query-len: {len(query)}")
                    #     print(f"dp_index_less_than_BSF={dp_index_less_than_BSF}")
                    #     print(f"ts_index_less_than_BSF={ts_index_less_than_BSF}")
                    bool_vec[ts_index_less_than_BSF] = False  # prune these indices

        # Get pruning rate
        PV = bool_vec[start_loc: N - subseq_len]
        if len(PV) > 0:
            PR = (len(PV)-sum(PV))/(len(PV))
        else:
            PR = 0
        # Get top discord
        discord_score = max(left_MP) - 0.000000001
        position = np.where(left_MP >= discord_score)[0]
        self.left_mp_result = left_MP
        self.position_found = position
        self.score_found = discord_score
        e_time = datetime.now()
        d_time = e_time - s_time
        if self.enable_output:
            print("Results:")
            print(f"DAMP_V2 run time {d_time}")
            print(f"Pruning Rate: {PR}")
            print(f"Predicted discord score/position: {discord_score} / {position}")
        return discord_score, position, left_MP, d_time

    def initial_checks(self, T, subsequence_len, start_loc):
        if self.enable_output:
            print("-----------------------------------------------")
            print("Thank you for using DAMP.")
            print("This is version 2.0 of DAMP, please email Eamonn Keogh (eamonn@cs.ucr.edu) " +
                  "or Yue Lu (ylu175@ucr.edu) to make sure you have the latest version.")
            print(f"This time series is of length {len(T)}, and the subsequence length is {subsequence_len}")
            la = self.lookahead
            self.lookahead = next_pow2(self.lookahead)
            print(f"The lookahead modified from {la} to the next power of 2: {self.lookahead}.")
            print("Hints:")
            print("Usually, the subsequence length you should use is between about 50 to 90% of a typical period.")
            print("A good initial value of lookahead is about 2^nearest_power_of_two(16 times S).")
            print("The range of lookahead should be 0 to length(T)-location_to_start_processing.")
            print("If speed is important, you can tune lookahead to get greater speed-up in your domain.")
            print("A simple search, doubling and halving the current value,")
            print(" should let you quickly converge on a good value.")
            print("------------------------------------------\n\n")
        if contains_constant_regions(T, subsequence_len):
            print("ERROR: ")
            print("This dataset contains constant and/or near constant regions.")
            print("We define the time series with an overall variance less than 0.2, ")
            print("or with a constant region within its sliding window as ")
            print("the time series containing constant and/or near constant regions.")
            print("Such regions can cause both false positives and ")
            print("false negatives depending on how you define anomalies.")
            print("And more importantly, it can also result in imaginary numbers in the ")
            print("calculated Left Matrix Profile, from which we cannot get the correct ")
            print("score value and position of the top discord.** The program has been terminated. **")
            sys.exit("This dataset contains constant and/or near constant regions.")
        if (start_loc/subsequence_len) < 4:
            print("WARNING: ")
            print("Location to start processing divided by SubsequenceLength is less than four.")
            print("We recommend that you allow DAMP to see at least four cycles,")
            print("otherwise you may get false positives early on.")
            print("If you have training data from the same domain, you can prepend the training data,")
            print("like this Data = [trainingdata, testdata], and call DAMP(data, S, length(trainingdata))")
            if start_loc < subsequence_len:
                print("Location_to_start_processing cannot be less than the subsequence length.")
                print(f"Location to start processing has been set to {subsequence_len}")
                self.start_loc = subsequence_len
            print("------------------------------------------\n\n")
        elif start_loc > (len(T)-subsequence_len+1):
            print("WARNING: ")
            print("Location to start processing cannot be greater than length(T)-S+1")
            self.start_loc = (len(T)-subsequence_len+1)
            print(f"Location to start processing has been set to {self.start_loc}.")
            print("------------------------------------------\n\n")
        if subsequence_len <= 8 or subsequence_len > 1000:  # <= 10 in original code
            old_len = subsequence_len
            autocor, lags = xcorr(T)
            subsequence_len = int(find_max_peak_index(autocor[3010:4001], lags[3010:4001]))
            print("WARNING: ")
            print("The subsequence length you set may be too large or too small")
            print(f"For the current input, we recommend setting the subsequence length to {subsequence_len}")
            print("------------------------------------------\n\n")


if __name__ == '__main__':
    pass
