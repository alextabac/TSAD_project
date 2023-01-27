
import numpy as np
from datetime import datetime
from MASS_V4 import MASS_V4


class DAMP_topK:
    def __init__(self, var_arg_in):
        self.varargin = dict(var_arg_in)
        self.lookahead = self.varargin["lookahead"]
        self.enable_output = self.varargin["enable_output"]
        self.start_loc = 0
        self.subseq_len = 0

    def DAMP_k(self, T, k_discords, start_loc):
        s_time = datetime.now()
        self.start_loc = start_loc
        self.subseq_len = subseq_len
        self.initial_checks(T, subseq_len, start_loc)
