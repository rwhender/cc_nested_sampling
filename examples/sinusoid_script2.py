# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:39:25 2015

@author: wesley

Sinusoid likelihood script
Copyright (C) 2015  R. Wesley Henderson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from cc_nested_sampling import NS, mh_oat, Loglikelihood
from cc_nested_sampling.combined_chain_ns import ParallelNS
from cc_nested_sampling.parallel_replace_ns import PReplaceNS
from numpy import matlib
import numpy as np
import dill
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import datetime
import os
from copy import deepcopy
import pdb


class SinLhood(Loglikelihood):
    def __init__(self, t, data, fs, Amin, Amax):
        n = data.size
        self.t = matlib.reshape(t, [1, n])
        self.data = matlib.reshape(data, [1, n])
        self.fs = fs
        self.Amin = Amin
        self.Amax = Amax
        self.fmin = 0.0
        self.fmax = fs/10

    def evaluate(self, theta, extra=False):
        for item in theta:
            if item < 0.0 or item > 1.0:
                return -np.inf
        sigma = 0.1
        m = theta.size / 3
        n = self.data.size
        A = (self.Amax - self.Amin) * theta[0::3] + self.Amin
        B = (self.Amax - self.Amin) * theta[1::3] + self.Amin
        f = (self.fmax - self.fmin) * theta[2::3] + self.fmin
        # i = np.arange(1, m + 1)[np.newaxis].T
        # f = fmax * np.cumprod(np.power(theta[2::3],
        #                                (1 / (m - i + 1))))[np.newaxis].T
        A = matlib.reshape(A, [m, 1])
        B = matlib.reshape(B, [m, 1])
        f = matlib.reshape(f, [m, 1])
        A = matlib.repmat(A, 1, n)
        B = matlib.repmat(B, 1, n)
        f = matlib.repmat(f, 1, n)
        t = matlib.repmat(self.t, m, 1)
        # data = matlib.repmat(data, m, 1)
        g = np.sum(A * np.cos(2 * np.pi * f * t) + B * np.sin(2 * np.pi *
                                                              f * t), 0)
        g = matlib.reshape(g, self.data.shape)
        logL = -1 * np.sum((g - self.data) ** 2) / (2 * sigma ** 2)
        if extra:
            return logL, A[:, 0], B[:, 0], f[:, 0], g
        else:
            return logL


def main():
    with open('sinusoid_data_20161004.dill', mode='rb') as f:
        data = dill.load(f)
        t = dill.load(f)
        _ = dill.load(f)
        fs = dill.load(f)
    N = 50
    MIN = 1
    MAX = 1e6
    lhood = SinLhood(t, data, fs, -2, 2)
    nsobj1 = ParallelNS(3, 4, lhood, verbose=False, N=N, explore=mh_oat,
                        MAX=MAX, MIN=MIN)
    nsobj2 = ParallelNS(6, 4, lhood, verbose=False, N=N, explore=mh_oat,
                        MAX=MAX, MIN=MIN)
    nsobj3 = ParallelNS(9, 4, lhood, verbose=False, N=N, explore=mh_oat,
                        MAX=MAX, MIN=MIN)
    nsobj4 = ParallelNS(12, 4, lhood, verbose=False, N=N, explore=mh_oat,
                        MAX=MAX, MIN=MIN)
    # explore = None
    start_time = time.time()
    logZ1, dead_samples1, H1, count1 = nsobj1.run()
    logZ2, dead_samples2, H2, count2 = nsobj2.run()
    logZ3, dead_samples3, H3, count3 = nsobj3.run()
    logZ4, dead_samples4, H4, count4 = nsobj4.run()
    end_time = time.time()
    plt.figure()
    plt.plot((2, 3, 4), (logZ2-logZ1, logZ3-logZ1, logZ4-logZ1), 'o')
    plt.xlabel('Model')
    plt.ylabel('log odds vs model 1')
    map1 = dead_samples1[-1].theta
    map2 = dead_samples2[-1].theta
    map3 = dead_samples3[-1].theta
    map4 = dead_samples4[-1].theta
    _, _, _, _, g1 = lhood.evaluate(map1, True)
    _, _, _, _, g2 = lhood.evaluate(map2, True)
    _, _, _, _, g3 = lhood.evaluate(map3, True)
    _, _, _, _, g4 = lhood.evaluate(map4, True)
    plt.figure()
    plt.plot(t, data, 'o', t, g1.T, t, g2.T, t, g3.T, t, g4.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.legend(('Data', 'g1', 'g2', 'g3', 'g4'))
    print("Running time is", end_time - start_time, "seconds")
    # mybounds = MyBounds(6)
    # x0 = np.random.rand(6)
    # res = basinhopping(sin_lhood, x0, 1500, accept_test=mybounds)
    # logL, A, B, f, g = sin_lhood(res.x, returns='params')
    # res = -1*sin_lhood(np.zeros(6))


if __name__ == "__main__":
    main()

