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

from nested_sampling import NS, mh_oat, Loglikelihood
from nested_sampling.combined_chain_ns import ParallelNS
from nested_sampling.parallel_replace_ns import PReplaceNS
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


# mat_dict = sio.loadmat('sinusoid_1-0_0-1_13-20_0.1_2.mat')
# outerdata = mat_dict['data']
# outertime = mat_dict['time']
# fs = mat_dict['fs']


class MyBounds(object):
    def __init__(self, d):
        self.xmax = np.ones(d)
        self.xmin = np.zeros(d)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


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


def serial_test(ntests):
    with open('sinusoid_data_20161004.dill', mode='rb') as f:
        data = dill.load(f)
        t = dill.load(f)
        dill.load(f)
        fs = dill.load(f)
    N = 20
    MIN = 1
    MAX = 1e6
    lhood = SinLhood(t, data, fs, -2, 2)
    foldername = ('Sinusoid Multitest result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    args = []
    nsobj1 = NS(3, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj2 = NS(6, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj3 = NS(9, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj4 = NS(12, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    for i in range(ntests):
        id1 = 'model1_' + str(i)
        id2 = 'model2_' + str(i)
        id3 = 'model3_' + str(i)
        id4 = 'model4_' + str(i)
        args.append((deepcopy(nsobj1), foldername, id1))
        args.append((deepcopy(nsobj2), foldername, id2))
        args.append((deepcopy(nsobj3), foldername, id3))
        args.append((deepcopy(nsobj4), foldername, id4))
    outputs = None
    with Pool() as pool:
        outputs = pool.starmap(serial_test_worker, args)
    logZs, Hs = ([], [], [], []), ([], [], [], [])
    for output in outputs:
        if 'model1' in output[1]:
            logZs[0].append(output[0])
            Hs[0].append(output[2])
        elif 'model2' in output[1]:
            logZs[1].append(output[0])
            Hs[1].append(output[2])
        elif 'model3' in output[1]:
            logZs[2].append(output[0])
            Hs[2].append(output[2])
        elif 'model4' in output[1]:
            logZs[3].append(output[0])
            Hs[3].append(output[2])
    writedict = {'logZs': logZs, 'Hs': Hs}
    with open(foldername + '/compiled_logZs_and_Hs.dill', 'wb') as f:
        dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
    return logZs, Hs


def parallel_test(ntests):
    with open('sinusoid_data_20161004.dill', mode='rb') as f:
        data = dill.load(f)
        t = dill.load(f)
        dill.load(f)
        fs = dill.load(f)
    N = 20
    M = 10
    MIN = 1
    MAX = 1e6
    lhood = SinLhood(t, data, fs, -2, 2)
    foldername = ('Sinusoid Multitest result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    nsobj1 = ParallelNS(3, M, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj2 = ParallelNS(6, M, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj3 = ParallelNS(9, M, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobj4 = ParallelNS(12, M, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat)
    nsobjs = (nsobj1, nsobj2, nsobj3, nsobj4)
    ids = ('model1_', 'model2_', 'model3_', 'model4_')
    logZs, Hs = ([], [], [], []), ([], [], [], [])
    for i in range(ntests):
        for idj, nsobj, logZ, H in zip(ids, nsobjs, logZs, Hs):
            idcur = idj + str(i)
            output = deepcopy(nsobj).run()
            writedict = {'logZs': output[0], 'samples': output[1],
                         'H': output[2], 'count': output[3]}
            with open(foldername + '/' + idcur + '.dill', 'wb') as f:
                dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
            logZ.append(output[0])
            H.append(output[2])
    with open(foldername + '/compiled_logZs_and_Hs.dill', 'wb') as f:
        dill.dump({'logZs': logZs, 'Hs': Hs}, f, dill.HIGHEST_PROTOCOL)
    return logZs, Hs


def parallel_replace_test(ntests):
    with open('sinusoid_data_20161004.dill', mode='rb') as f:
        data = dill.load(f)
        t = dill.load(f)
        dill.load(f)
        fs = dill.load(f)
    N = 400
    R = 4
    MIN = 1
    MAX = 1e6
    lhood = SinLhood(t, data, fs, -2, 2)
    foldername = ('Sinusoid PReplace Multitest result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    nsobj1 = PReplaceNS(3, R, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat,
                        verbose=False)
    nsobj2 = PReplaceNS(6, R, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat,
                        verbose=False)
    nsobj3 = PReplaceNS(9, R, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat,
                        verbose=False)
    nsobj4 = PReplaceNS(12, R, lhood, N=N, MIN=MIN, MAX=MAX, explore=mh_oat,
                        verbose=False)
    nsobjs = (nsobj1, nsobj2, nsobj3, nsobj4)
    ids = ('model1_', 'model2_', 'model3_', 'model4_')
    logZs, Hs = ([], [], [], []), ([], [], [], [])
    for i in range(ntests):
        for idj, nsobj, logZ, H in zip(ids, nsobjs, logZs, Hs):
            idcur = idj + str(i)
            output = deepcopy(nsobj).run()
            writedict = {'logZs': output[0], 'samples': output[1],
                         'H': output[2], 'count': output[3]}
            with open(foldername + '/' + idcur + '.dill', 'wb') as f:
                dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
            logZ.append(output[0])
            H.append(output[2])
    with open(foldername + '/compiled_logZs_and_Hs.dill', 'wb') as f:
        dill.dump({'logZs': logZs, 'Hs': Hs}, f, dill.HIGHEST_PROTOCOL)
    return logZs, Hs


def serial_test_worker(nsobj, foldername, result_id):
    output = nsobj.run()
    writedict = {'logZs': output[0], 'samples': output[1], 'H': output[2],
                 'count': output[3]}
    with open(foldername + '/' + result_id + '.dill', 'wb') as f:
        dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
    return (output, result_id)


def evidence_plotter(logZs):
    """Plot the log-evidence for models, assuming a break is necessary between
    1 and 2.

    Modeled after this stackoverflow answer:
    http://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib

    Parameters
    ----------
    logZs : list
        list of four lists or arrays of log-evidence values
    """
    ymin1 = np.min(logZs[0]) - 5
    ymax1 = np.max(logZs[0]) + 5
    ymin2 = np.min(logZs[3]) - 5
    ymax2 = np.max(logZs[1]) + 5
    f = plt.figure(facecolor='w')
    # Add subplot that covers entire figure, so we can use good labels
    ax = f.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
    ax.set_ylabel('Model logZ')
    ax.set_xlabel('Number of sinusoids in model')
    ax.yaxis.labelpad = 20
    # (ax2, ax1) = plt.subplots(2, 1, sharex=True, facecolor='w')
    ax1 = f.add_subplot(212)
    ax2 = f.add_subplot(211)
    ax1.boxplot(logZs)
    ax2.boxplot(logZs)
    ax1.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2, ymax2)
    ax1.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_bottom()
    ax1.tick_params(labeltop='off')
    ax2.xaxis.tick_top()
    d = 0.015  # Tick length
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1-d, 1+d), (-d, +d), **kwargs)
    plt.show()


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
    # main()
    logZs, Hs = parallel_test(20)
    # logZs, Hs = serial_test(20)
    evidence_plotter(logZs)
    # pass
