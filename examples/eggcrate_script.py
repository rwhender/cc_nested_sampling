# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 08:25:23 2016

@author: wesley

Eggcrate likelihood script
Copyright (C) 2016  R. Wesley Henderson

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

from nested_sampling import mh, Loglikelihood, importance_resampling
from nested_sampling.combined_chain_ns import ParallelNS
import numpy as np
import dill
import matplotlib.pyplot as plt
import datetime
import os
from copy import deepcopy
import pdb


# mat_dict = sio.loadmat('sinusoid_1-0_0-1_13-20_0.1_2.mat')
# outerdata = mat_dict['data']
# outertime = mat_dict['time']
# fs = mat_dict['fs']


class EggcrateLhood(Loglikelihood):
    def __init__(self):
        pass

    def evaluate(self, theta, extra=False):
        for item in theta:
            if item < 0.0 or item > 1.0:
                return -np.inf
        maxval = 0.0
        minval = 10.0 * np.pi
        x = (maxval - minval) * theta[0] + minval
        y = (maxval - minval) * theta[1] + minval
        logL = (2 + np.cos(x/2) * np.cos(y/2)) ** 5
        if extra:
            return logL, x, y
        else:
            return logL


def parallel_test(ntests):
    N = 16
    M = 20
    MIN = 1
    MAX = 1e6
    lhood = EggcrateLhood()
    foldername = ('Eggcrate Multitest result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    nsobj = ParallelNS(2, M, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh)
    idj = 'model1_'
    logZs, Hs = [], []
    for i in range(ntests):
        idcur = idj + str(i)
        output = deepcopy(nsobj).run()
        writedict = {'logZs': output[0], 'samples': output[1],
                     'H': output[2], 'count': output[3]}
        with open(foldername + '/' + idcur + '.dill', 'wb') as f:
            dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
        logZs.append(output[0])
        Hs.append(output[2])
    with open(foldername + '/compiled_logZs_and_Hs.dill', 'wb') as f:
        dill.dump({'logZs': logZs, 'Hs': Hs}, f, dill.HIGHEST_PROTOCOL)
    return logZs, Hs


def plot_hist(samples):
    def logmod(x):
        logx = -241.76 * np.ones(x.shape)
        logx[x != 0.0] = np.log(x[x != 0.0])
        return logx
    new_samples = importance_resampling(samples)
    thetas = [sample.theta for sample in new_samples]
    thetas = np.array(thetas)
    to_close = plt.figure()
    counts, x, y, _ = plt.hist2d(thetas[:, 0], thetas[:, 1], bins=[20, 20])
    plt.close(to_close)
    counts = counts / counts.sum().sum()
    X, Y = np.meshgrid(x, y)
    X, Y = 10 * np.pi * X, 10 * np.pi * Y
    plt.figure()
    plt.pcolormesh(X, Y, logmod(counts), vmax=0.2254, vmin=-241.76)
    plt.colorbar()
    plt.set_cmap('copper')
    plt.xlabel(r'$\Theta_1$')
    plt.ylabel(r'$\Theta_2$')


def plot_logp():
    lhood = EggcrateLhood()
    u = np.arange(0, 1, 0.5/(10*np.pi))
    v = np.arange(0, 1, 0.5/(10*np.pi))
    logp = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            logp[i, j] = lhood.evaluate(np.hstack((u[i], v[j])))
    logp = logp - 235.88 - 2 * np.log(10 * np.pi)
    print('Minimum:', logp.min())
    print('Maximum:', logp.max())
    x = np.arange(0, 10*np.pi, 0.5)
    y = np.arange(0, 10*np.pi, 0.5)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    plt.pcolormesh(X, Y, logp)
    plt.colorbar()
    plt.set_cmap('copper')
    plt.xlabel(r'$\Theta_1$')
    plt.ylabel(r'$\Theta_2$')
    return fig


if __name__ == "__main__":
    logZs, Hs = parallel_test(20)
    pass
