
import numpy as np
import pandas as pd
import random
from datetime import datetime


class HOT_SAX:
    def __init__(self, ts):
        self.alpha = int(5)  # default value
        self.wsize = int(8)  # default value
        self.alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        beta3 = [-0.43, 0.43]
        beta4 = [-0.67, 0.0, 0.67]
        beta5 = [-0.84, -0.25, 0.25, 0.84]
        beta6 = [-0.97, -0.43, 0.0, 0.43, 0.97]
        beta7 = [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07]
        beta8 = [-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15]
        beta9 = [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]
        beta10 = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        self.betas = [None, None, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10]
        self.beta = None  # the specific beta from betas above
        self.brp = None  # the break points
        self.sax_array = None
        self.sax_trie = {}
        self.sax_mindd = None
        self.ts = pd.DataFrame(data=ts, columns=['value'])
        self.idx = []  # outer heuristic order of indices
        self.set_alphabeta()
        self.best_dist = 0.0
        self.best_loc = -1

    def search(self, print_out=False):
        s_time = datetime.now()
        best_dist = 0.0
        best_loc = -1
        cnt = 500
        if print_out:
            print("HOT SAX search starting now.")
        for p in self.idx:
            nearest_neighbor_dist = np.Inf
            word = self.sax_array.loc[p, 'word']
            i_list = self.get_trie_list(word)
            dlist = i_list + self.idx
            for q in dlist:
                if abs(p - q) >= self.wsize:
                    dist = self.get_mindist(p, q)
                    if dist < best_dist:
                        break
                    if dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = dist
            if np.Inf > nearest_neighbor_dist > best_dist:
                best_dist = nearest_neighbor_dist
                best_loc = p
            if print_out:
                cnt -= 1
                if cnt < 0:
                    cnt = 500
                    print(f"Passed another 500 iterations out of {len(self.idx)}")
        self.best_dist = best_dist
        self.best_loc = best_loc
        e_time = datetime.now()
        d_time = e_time - s_time
        if print_out:
            print(f"HOT SAX completed with run time {d_time}.")
            print(f"Discord found at index {best_loc} with distance {best_dist}.")
        return best_dist, best_loc

    def init_norm(self):
        w = self.wsize
        n = len(self.ts) - w
        m = np.mean(self.ts['value'].values)
        s = np.std(self.ts['value'].values)
        if s > 0.0:
            self.ts['value'] = (self.ts['value'] - m) / s
        else:
            self.ts['value'] = (self.ts['value'] - m)
        self.ts['SAX'] = self.ts.apply(lambda r: self.get_sax(r.value), axis=1)
        # counting words
        sax_wc = {}
        for i in range(n):
            word = "".join(self.ts[i: (i+w)]['SAX'].values)
            if word in sax_wc:
                sax_wc[word] += 1
            else:
                sax_wc[word] = 1
        # list of dicts: index, word, and the word repeat count from above loop
        rows_l = []
        for i in range(n):
            word = "".join(self.ts[i: (i+w)]['SAX'].values)
            rows_l.append({'idx': i, 'word': word, 'count': sax_wc[word]})
        self.sax_array = pd.DataFrame(rows_l)
        self.sax_trie = {}
        for i, row in self.sax_array.iterrows():
            cc = self.sax_trie
            for c in row.word[:-1]:
                if c not in cc:
                    cc[c] = {}
                cc = cc[c]
            if row.word[-1] in cc:
                cc[row.word[-1]].append(row.idx)
            else:
                cc[row.word[-1]] = [row.idx]
        min_count = np.min(self.sax_array['count'].values)
        min_idx = []
        rest_idx = []
        for i, row in self.sax_array.iterrows():
            if row['count'] == min_count:
                min_idx.append(i)
            else:
                rest_idx.append(i)
        random.shuffle(rest_idx)
        self.idx = min_idx + rest_idx

    def get_mindist(self, p, q):
        word1 = self.sax_array.loc[p, 'word']
        word2 = self.sax_array.loc[q, 'word']
        return self.get_mindist_words(word1, word2)

    def get_mindist_words(self, word1, word2):
        r = self.sax_mindd[word1[0]][word2[0]] ** 2
        for i in range(1, self.alpha):
            r += self.sax_mindd[word1[i]][word2[i]] ** 2
        return np.sqrt(r / self.alpha)

    def get_trie_list(self, word):
        cc = self.sax_trie
        for c in word:
            cc = cc[c]
        return cc  # perhaps need to be cc.copy()

    def get_sax(self, v):
        for i in range(self.alpha-1):
            if v <= self.brp[i]:
                return self.alphabet[i]
        return self.alphabet[self.alpha-1]

    def set_SAX_params(self, alpha, word_size):
        self.alpha = int(alpha)      # SAX vocabulary size
        self.wsize = int(word_size)  # SAX word size
        self.set_alphabeta()

    def set_alphabeta(self):
        self.beta = self.betas[self.alpha]
        self.brp = {i: self.beta[i] for i in range(self.alpha - 1)}
        mind_mat = np.zeros((self.alpha, self.alpha))
        mdd = {}
        for i in range(self.alpha):
            for j in range(self.alpha):
                if abs(i-j) > 1:
                    mind_mat[i][j] = self.beta[max(i, j)-1] - self.beta[min(i, j)]
                if self.alphabet[i] not in mdd:
                    mdd[self.alphabet[i]] = {}
                mdd[self.alphabet[i]][self.alphabet[j]] = mind_mat[i][j]
        self.sax_mindd = mdd
