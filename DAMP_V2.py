
import numpy as np
from math import floor
from MASS_V4 import MASS_V4

class DAMP_V2:
    def __init__(self, var_arg_in):
        self.varargin = dict(var_arg_in)
        self.lookahead = self.varargin["lookahead"]
        self.enable_output = self.varargin["enable_output"]
        self.start_loc = 0
        self.subseq_len = 0

    def DAMP_2_0(self, T, subseq_len, start_loc):
        self.start_loc = start_loc
        self.subseq_len = subseq_len
        self.initial_checks(T, subseq_len, start_loc)
        mass_v4 = MASS_V4()
        start_loc = self.start_loc
        subseq_len = self.subseq_len
        left_MP = np.zeros(len(T))
        best_so_far = -np.Inf
        bool_vec = np.ones(len(T), dtype=bool)
        for i in range(start_loc, (start_loc + (16 * subseq_len))):
            # Skip the current iteration if the corresponding boolean value
            if not bool_vec[i]:
                left_MP[i] = left_MP[i-1]-0.00001
                continue
            if (i + subseq_len - 1) >= len(T):
                break
            query = T[i:(i+subseq_len)]
            left_MP[i] = min(mass_v4.get_similarities(T[:i+1], query))


    def next_pow2(self, x):
        # 1 if x == 0 else 2 ** (x - 1).bit_length()  # but no need to worry about x==0
        return 2 ** (x - 1).bit_length()

    def contains_constant_regions(self, T, subsequence_len):
        # Assuming T has reset index
        bool_vec = False
        constant_indices = np.ediff1d(np.where(np.ediff1d(T) == 0))
        serial_indices = np.split(constant_indices, np.where(constant_indices > 1)[0])
        lengths = [len(arr) for arr in serial_indices]
        max_len = max(lengths)
        if max_len >= subsequence_len or np.var(T) < 0.2:
            bool_vec = True
        return bool_vec

    def initial_checks(self, T, subsequence_len, start_loc):
        if self.enable_output:
            print("-----------------------------------------------")
            print("Thank you for using DAMP.")
            print("This is version 2.0 of DAMP, please email Eamonn Keogh (eamonn@cs.ucr.edu) " +
                  "or Yue Lu (ylu175@ucr.edu) to make sure you have the latest version.")
            print(f"This time series is of length {len(T)}, and the subsequence length is {subsequence_len}")
            la = self.lookahead
            self.lookahead = self.next_pow2(self.lookahead)
            print(f"The lookahead modified from {la} to the next power of 2: {self.lookahead}.")
            print("Hints:")
            print("Usually, the subsequence length you should use is between about 50 to 90% of a typical period.")
            print("A good initial value of lookahead is about 2^nearest_power_of_two(16 times S).")
            print("The range of lookahead should be 0 to length(T)-location_to_start_processing.")
            print("If speed is important, you can tune lookahead to get greater speed-up in your domain.")
            print("A simple search, doubling and halving the current value,")
            print(" should let you quickly converge on a good value.")
            print("------------------------------------------\n\n")
        if self.contains_constant_regions(T, subsequence_len):
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
            quit()
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
            print("WARNING: ");
            print("Location to start processing cannot be greater than length(T)-S+1")
            self.start_loc = (len(T)-subsequence_len+1)
            print(f"Location to start processing has been set to {self.start_loc}.")
            print("------------------------------------------\n\n")
        if subsequence_len <= 8 or subsequence_len > 1000:  # <= 10 in original code
            # [autocor,lags] = xcorr(T,3000,'coeff');
            # [~,ReferenceSubsequenceLength] = findpeaks(autocor(3010:4000),lags(3010:4000),'SortStr','descend','NPeaks',1);
            # self.subseq_len(isempty(ReferenceSubsequenceLength))=1000;
            # self.subseq_len = floor(self.subseq_len)
            print("ERROR: ")  # should be warning if able to find automatic - fix code above
            print("The subsequence length you set may be too large or too small.")
            # print(f"For the current input T, we recommend setting the subsequence length to {self.subseq_len}")
            print("For the current input T, we recommend setting the subsequence to other value, now quit.")
            print("------------------------------------------\n\n")
            quit()

