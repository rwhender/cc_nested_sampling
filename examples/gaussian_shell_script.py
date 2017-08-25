# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 14:47:00 2017

@title: Gaussian shell problem
@author: wesley

Gaussian shell likelihood script
Copyright (C) 2017  R. Wesley Henderson

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

from nested_sampling import NS, Loglikelihood, importance_resampling, mh
from nested_sampling.combined_chain_ns import ParallelNS
from numpy import matlib
import numpy as np
import dill
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import datetime
import os
from copy import deepcopy, copy
import pdb
import sys
from scipy.special import erf


def log_plus(x, y):
    """Perform log-addition while maintaining maximum precision.

    Parameters
    ----------
    x : float
        first value to add
    y : float
        second value to add

    Returns
    -------
    float
        log-sum of x and y
    """
    if x > y:
        return x + np.log(1+np.exp(y-x))
    else:
        return y + np.log(1+np.exp(x-y))


class GShellLhood(Loglikelihood):
    """Likelihood object for single Gaussian shell"""
    def __init__(self, clist, rlist, wlist):
        self.n = len(clist)
        self.m = len(clist[0])
        self.clist = clist
        self.rlist = rlist
        self.wlist = wlist
        self.thetamax = 6.0
        self.thetamin = -6.0

    def evaluate(self, theta, extra=False):
        for item in theta:
            if item < 0.0 or item > 1.0:
                return -np.inf
        # Scale theta properly
        thetanew = copy(theta)
        thetanew = ((self.thetamax - self.thetamin) * thetanew +
                    self.thetamin)
        logL = -np.inf
        for c, r, w in zip(self.clist, self.rlist, self.wlist):
            tmc = thetanew - c
            val = (-np.log(np.sqrt(2*np.pi)*w) -
                   ((np.sqrt(tmc.dot(tmc)) - r)**2 / (2 * w**2)))
            logL = log_plus(logL, val)
        if extra:
            return logL, thetanew
        else:
            return logL


class GShellLhood2(Loglikelihood):
    """Likelihood object for two Gaussian shells"""
    def __init__(self, c1, c2, r1, r2, w1, w2):
        self.n = 2
        self.m = len(c1)
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.w1 = w1
        self.w2 = w2
        self.thetamax = 6.0
        self.thetamin = -6.0

    def evaluate(self, theta, extra=False, pieces=False):
        for item in theta:
            if item < 0.0 or item > 1.0:
                return -np.inf
        # Scale theta properly
        thetanew = copy(theta)
        thetanew = ((self.thetamax - self.thetamin) * thetanew +
                    self.thetamin)
        c1, c2, r1, r2, w1, w2 = (self.c1, self.c2, self.r1, self.r2, self.w1,
                                  self.w2)
        tmc1 = thetanew - c1
        tmc2 = thetanew - c2
        val1 = (-np.log(np.sqrt(2*np.pi)*w1) -
                ((np.sqrt(tmc1.dot(tmc1)) - r1)**2 / (2 * w1**2)))
        val2 = (-np.log(np.sqrt(2*np.pi)*w2) -
                ((np.sqrt(tmc2.dot(tmc2)) - r2)**2 / (2 * w2**2)))
        logL = log_plus(val1, val2)
        if not extra and not pieces:
            returns = logL
        else:
            returns = [logL]
        if extra:
            returns.append(thetanew)
        if pieces:
            returns.append(val1)
            returns.append(val2)
        return returns

    def grad(self, theta):
        """Compute the gradient of the log-likelihood."""
        if theta[0] <= 0.5:
            c, r, w = self.c1, self.r1, self.w1
        else:
            c, r, w = self.c2, self.r2, self.w2
        A = self.thetamax - self.thetamin
        B = self.thetamin
        tmc = A * theta + B - c
        g = (-(1/w**2) * (np.sqrt(tmc.dot(tmc)) - r) *
             (1/np.sqrt(tmc.dot(tmc))) * tmc * A)
        return g


class GLhood2(Loglikelihood):
    """Assume uncorrelated, and that mu and sigma are vectors."""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.thetamax = 6.0
        self.thetamin = -6.0
        self.m = len(mu)

    def evaluate(self, theta, extra=False):
        x = ((self.thetamax - self.thetamin) * theta +
             self.thetamin)[np.newaxis].T
        x_mu = x - self.mu
        k = -(self.m/2)*np.log(2 * np.pi) - np.log(self.sigma).sum()
        logL = k - (1/2) * ((x_mu / self.sigma)**2).sum()
        if not extra:
            return logL
        else:
            return logL, x

    def grad(self, theta):
        A = self.thetamax - self.thetamin
        B = self.thetamin
        g = -(1/self.sigma**2) * (A * theta + B - self.mu) * A
        return g


def serial_test(ntests):
    """Use multiprocessing.Pool to do ntests serial NS runs in parallel"""
    ndim = 30
    N = int(10*ndim + 0.5)
    MIN = 1
    MAX = 1e7
    clist = [np.zeros(ndim), np.zeros(ndim)]
    clist[0][0] = -3.5
    clist[1][0] = 3.5
    wlist = [0.1, 0.1]
    rlist = [2.0, 2.0]
    lhood = GShellLhood2(clist[0], clist[1], rlist[0], rlist[1], wlist[0],
                         wlist[1])
    nsobj = NS(ndim, lhood, verbose=False, N=N,
               explore=mh, MAX=MAX, MIN=MIN)
    foldername = ('Gaussian Shell 30 Multitest S result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    args = []
    for i in range(ntests):
        id1 = 'gs30_M_1_N_' + str(N) + '_' + str(i)
        args.append((deepcopy(nsobj), foldername, id1))
    outputs = None
    with Pool() as pool:
        outputs = pool.starmap(serial_test_worker, args)
    logZs, Hs = [], []
    for output in outputs:
        logZs.append(output[0][0])
        Hs.append(output[0][2])
    writedict = {'logZs': logZs, 'Hs': Hs}
    with open(foldername + '/compiled_logZs_and_Hs.dill', 'wb') as f:
        dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
    return logZs, Hs


def parallel_test(ntests):
    """Do ntests parallel NS runs one at a time"""
    ndim = 30
    M = 4
    N = int(10*ndim/M + 0.5)
    MIN = 1
    MAX = 1e7
    clist = [np.zeros(ndim), np.zeros(ndim)]
    clist[0][0] = -3.5
    clist[1][0] = 3.5
    wlist = [0.1, 0.1]
    rlist = [2.0, 2.0]
    lhood = GShellLhood2(clist[0], clist[1], rlist[0], rlist[1], wlist[0],
                         wlist[1])
    nsobj = ParallelNS(ndim, M, lhood, verbose=False, N=N,
                       explore=mh, MAX=MAX, MIN=MIN)
    foldername = ('Gaussian Shell 30 Multitest P result {:%Y-%m-%d %H.%M.%S}'.
                  format(datetime.datetime.now()))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    id1 = 'gs30_M_' + str(M) + '_N_' + str(N) + '_'
    logZs, Hs = [], []
    for i in range(ntests):
        idcur = id1 + str(i)
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


def serial_test_worker(nsobj, foldername, result_id):
    """Worker for multiprocessing.Pool in serial_test"""
    output = nsobj.run()
    writedict = {'logZs': output[0], 'samples': output[1], 'H': output[2],
                 'count': output[3]}
    with open(foldername + '/' + result_id + '.dill', 'wb') as f:
        dill.dump(writedict, f, dill.HIGHEST_PROTOCOL)
    return (output, result_id)


def main():
    # Log-evidences for reference
    # 2: -1.75
    # 5: -5.67
    # 10: -14.59
    # 20: -36.09
    # 30: -60.13
    ndim = 100
    M = 1
    N = int(10*ndim/M + 0.5)
    # N = 1000
    MIN = 1
    MAX = 1e7
    clist = [np.zeros(ndim), np.zeros(ndim)]
    clist[0][0] = -3.5
    clist[1][0] = 3.5
    wlist = [0.1, 0.1]
    rlist = [2.0, 2.0]
    lhood = GShellLhood2(clist[0], clist[1], rlist[0], rlist[1], wlist[0],
                         wlist[1])
    # nsobj1 = ParallelNS(ndim, M, lhood, verbose=False, N=N,
    #                     explore=gmc.mixed_explore, MAX=MAX, MIN=MIN)
    nsobj1 = NS(ndim, lhood, verbose=True, N=N,
                explore=mh, MAX=MAX, MIN=MIN)
    # explore = None
    start_time = time.time()
    logZ1, dead_samples1, H1, count1 = nsobj1.run()
    end_time = time.time()
    # map1 = dead_samples1[-1].theta
    # _, theta = lhood.evaluate(map1, True)
    print("Running time is", end_time - start_time, "seconds")
    try:
        samples = importance_resampling(dead_samples1)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print("Don't trust the samples!")
        return logZ1, dead_samples1, H1, count1
    return logZ1, samples, H1, count1


def lhood_test():
    clist = [np.array([-3.5, 0]), np.array([3.5, 0])]
    wlist = [0.1, 0.1]
    rlist = [2.0, 2.0]
    lhood = GShellLhood2(clist[0], clist[1], rlist[0], rlist[1], wlist[0],
                         wlist[1])
    xgrid = np.arange(0.0, 1.0, 1/400)
    ygrid = np.arange(0.0, 1.0, 1/400)
    X, Y = matlib.meshgrid(xgrid, ygrid)
    logL = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            logL[i, j] = lhood.evaluate(theta)
    plt.pcolormesh(12*X-6, 12*Y-6, np.exp(logL))
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    return X, Y, logL


def logZ_grid(delta=1/400):
    clist = [np.array([-3.5, 0]), np.array([3.5, 0])]
    wlist = [0.1, 0.1]
    rlist = [2.0, 2.0]
    lhood = GShellLhood(clist, rlist, wlist)
    log_delta = np.log(delta)
    xgrid = np.arange(0.0, 1.0, delta)
    ygrid = np.arange(0.0, 1.0, delta)
    logL = np.zeros((len(xgrid), len(ygrid)))
    logZ = -np.inf
    for i in range(logL.shape[0]):
        for j in range(logL.shape[1]):
            theta = np.array([xgrid[i], ygrid[j]])
            logL[i, j] = lhood.evaluate(theta)
            logZ = log_plus(logZ, logL[i, j] + 2 * log_delta)
    return logZ


def main_gauss2():
    ndim = 2
    M = 1
    N = int(100*ndim/M + 0.5)
    MIN = 1
    MAX = 1e7
    mu = np.zeros(ndim)
    sigma = np.ones(ndim)
    lhood = GLhood2(mu, sigma)
    nsobj1 = NS(ndim, lhood, verbose=True, N=N,
                explore=mh, MAX=MAX, MIN=MIN)
    start_time = time.time()
    logZ1, dead_samples1, H1, count1 = nsobj1.run()
    end_time = time.time()
    print("Running time is", end_time - start_time, "seconds")
    try:
        samples = importance_resampling(dead_samples1)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print("Don't trust the samples!")
        return logZ1, dead_samples1, H1, count1
    return logZ1, samples, H1, count1


def gcdf(x, mu, sigma):
    return (1/2) * (1 + erf((x - mu)/(sigma*np.sqrt(2))))


def logZ_grid_g2(delta=1/400):
    ndim = 2
    mu = np.zeros(ndim)
    sigma = np.ones(ndim)
    lhood = GLhood2(mu, sigma)
    log_delta = np.log(delta)
    xgrid = np.arange(0.0, 1.0, delta)
    ygrid = np.arange(0.0, 1.0, delta)
    logL = np.zeros((len(xgrid), len(ygrid)))
    logZ = -np.inf
    for i in range(logL.shape[0]):
        for j in range(logL.shape[1]):
            theta = np.array([xgrid[i], ygrid[j]])
            logL[i, j] = lhood.evaluate(theta)
            logZ = log_plus(logZ, logL[i, j] + 2 * log_delta)
    return logZ


if __name__ == "__main__":
    # main()
    # logZs, Hs = parallel_test(20)
    # logZs, Hs = serial_test(20)
    # evidence_plotter(logZs)
    # out = main_gauss2()
    # out = main()
    # print('logZ =', out[0])
    # samples = out[1]
    # thetas = [sample.theta for sample in samples]
    # thetas = np.vstack(thetas)
    # plt.figure()
    # ax = plt.axes(xlim=[0, 1], ylim=[0, 1])
    # plt.hist2d(thetas[:, 0], thetas[:, 1], bins=20)
    # plt.colorbar()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    out_serial = serial_test(20)
    out_parallel = parallel_test(20)
