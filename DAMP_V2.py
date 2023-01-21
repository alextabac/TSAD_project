
import numpy as np

class DAMP_V2:

    def contains_constant_regions(self, T, subsequence_len):
        bool_vec = False
        constant_indices = np.ediff1d(np.where(np.ediff1d(T) == 0))
        serial_indices = np.split(constant_indices, np.where(constant_indices > 1)[0])
        lengths = [len(arr) for arr in serial_indices]
        max_len = max(lengths)
        if max_len >= subsequence_len or np.var(T) < 0.2:
            bool_vec = True
        return bool_vec
