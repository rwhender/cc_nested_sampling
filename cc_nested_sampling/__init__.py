# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:21:41 2015

@title: Nested sampling
@author: Wesley

Nested sampling
Copyright (C) 2015  R. Wesley Henderson

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser GNU General Public License for more details.

You should have received a copy of the Lesser GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numpy import random
import numpy as np
from operator import attrgetter
from copy import deepcopy, copy
from numpy.matlib import repmat


def mh(samples, L, idx_min, lhood):
    """
    Explore the prior constrained by L

    Parameters
    ----------
    samples : list
        list of Sample objects
    L : float
        log-likelihood threshold
    idx_min : int
        index of minimum likelihood sample
    lhood : class instance
        log-likelihood class instance

    Returns
    -------
    sample : Sample object
        updated sample
    """
    random.seed()
    N = len(samples)
    clone_idx = random.randint(N)
    current = deepcopy(samples[clone_idx])
    P = (current.theta).size
    step = 0.01
    T = 2000
    accept = 0
    reject = 0

    for i in range(T):
        trial = current.theta + step * random.randn(P)
        trial = trial - np.floor(trial)
        logL_trial = lhood.evaluate(trial)
        if logL_trial >= L:
            accept += 1
            current.theta = deepcopy(trial)
            current.logL = copy(logL_trial)
        else:
            reject += 1

        if accept > reject:
            step *= np.exp(2/accept)
        elif accept < reject:
            step /= np.exp(2/reject)
    return current


def mh_oat(samples, L, idx_min, lhood):
    """
    Explore the prior constrained by L by varying each dimension separately

    Parameters
    ----------
    samples : list
        list of Sample objects
    L : float
        log-likelihood threshold
    idx_min : int
        index of minimum likelihood sample
    lhood : class instance
        log-likelihood class instance

    Returns
    -------
    sample : Sample object
        updated sample
    """
    random.seed()
    N = len(samples)
    clone_idx = random.randint(N)
    current = deepcopy(samples[clone_idx])
    P = (current.theta).size
    step = 0.01 * np.ones(current.theta.shape)
    T = 1000
    accept = np.zeros(current.theta.shape)
    reject = np.zeros(current.theta.shape)
    trial = deepcopy(current.theta)

    for i in range(T):
        for p in random.permutation(P):
            trial[p] = current.theta[p] + step[p] * random.randn()
            trial = trial - np.floor(trial)
            logL_trial = lhood.evaluate(trial)
            if logL_trial >= L:
                accept[p] += 1
                current.theta = deepcopy(trial)
                current.logL = copy(logL_trial)
            else:
                reject[p] += 1

            if accept[p] > reject[p]:
                step[p] *= np.exp(2/accept[p])
            elif accept[p] < reject[p]:
                step[p] /= np.exp(2/reject[p])
    return current


class Sample:
    """Initialize a random sample, with a likelihood

    Parameters
    ----------
    P : int
        Number of parameters
    lhood : class instance
        log-likelihood class instance
    """
    def __init__(self, P, lhood):
        self.theta = random.rand(P)
        self.logL = lhood.evaluate(self.theta)
        self.logWt = -np.Inf

    # Define ordering operations, so that samples can be compared by logL
    def __lt__(self, x):
        return self.logL < x.logL

    def __le__(self, x):
        return self.logL <= x.logL

    def __eq__(self, x):
        return self.logL == x.logL

    def __ne__(self, x):
        return self.logL != x.logL

    def __gt__(self, x):
        return self.logL > x.logL

    def __ge__(self, x):
        return self.logL >= x.logL

    # Define a distance function to allow for clustering
    def distance(self, x):
        diff = self.theta - x.theta
        return np.sqrt(diff.dot(diff))


class NS:
    """
    Initialize the NS nested sampling object.

    Parameters
    ----------
    P : int
        number of model parameters to pass to log-likelihood function
    lhood : class instance
        class instance of the form ns.Loglikelihood
    explore : function handle, optional
        function handle of form new_sample = explore(live_samples,
        logL_min, idx_min, lhood). Uses built-in Metropolis algorithm by
        default
    N : int, optional
        number of live samples. Default is 20
    MIN : int, optional
        minimum number of likelihood constraint iterations. Default is 1
    MAX : int, optional
        maximum number of likelihood constraint iterations. Default is 1e5
    verbose : bool, optional
        If true, print status messages at each iteration. Default is False

    """
    def __init__(self, P, lhood, explore=mh, N=20, MIN=1, MAX=1e6,
                 verbose=False, *args, **kwargs):
        self.P = P
        self.lhood = lhood
        self.explore = explore
        self.N = N
        self.MIN = MIN
        self.MAX = MAX
        self.verbose = verbose

    def run(self):
        """Run nested sampling algorithm

        Returns
        -------
        logZ : float
            esimated log-evidence
        dead_samples : list of Samples
            list of discarded Sample objects
        H : float
            estimated information
        count : int
            number of likelihood constraint iterations
        """
        return self.run_serial()

    def run_serial(self):
        """Run nested sampling algorithm

        Returns
        -------
        logZ : float
            esimated log-evidence
        dead_samples : list of Samples
            list of discarded Sample objects
        H : float
            estimated information
        count : int
            number of likelihood constraint iterations
        """
        self.logZ = -np.inf
        self.H = np.float64(0.0)
        log_width = np.log(1-np.exp(-1/self.N))
        live_samples = [Sample(self.P, self.lhood) for i in range(self.N)]
        self.dead_samples = []
        iterate = True
        self.count = 0
        while iterate:
            # min_sample = min(live_samples, key=attrgetter('logL'))
            min_sample = min(live_samples)
            logL_min = min_sample.logL
            if self.verbose:
                print("P =", self.P, " count =", self.count, " L =", logL_min)
            idx_min = live_samples.index(min_sample)
            min_sample.logWt = logL_min + log_width
            logZnew = self.log_plus(self.logZ, min_sample.logWt)
            H_new = (np.exp(min_sample.logWt - logZnew)*logL_min +
                     np.exp(self.logZ - logZnew)*(self.H + self.logZ) -
                     logZnew)
            if ~np.isnan(H_new):
                self.H = H_new
            self.logZ = logZnew
            min_sample = deepcopy(min_sample)
            live_samples[idx_min] = deepcopy(self.explore(live_samples,
                                                          logL_min,
                                                          idx_min, self.lhood))
            self.dead_samples.append(deepcopy(min_sample))
            log_width -= 1/self.N
            self.count += 1
            if self.count > self.MIN and self.count > 3 * self.H * self.N:
                iterate = False
            elif self.count >= self.MAX:
                iterate = False
        if self.verbose:
            print("Done!")
        return self.logZ, self.dead_samples, self.H, self.count

    def set_params(self, *args, **kwargs):
        pass

    def get_params(self, verbose=False):
        """Return and optionally print nested sampling parameters

        Parameters
        ----------
        verbose : bool, optional
            Default False. If True, prints results to terminal

        Returns
        -------
        P : int
            number of model parameters to pass to log-likelihood function
        N : int
            number of live samples
        MIN : int
            minimum number of likelihood constraint iterations
        MAX : int
            maximum number of likelihood constraint iterations
        """
        if verbose:
            print("P =", self.P)
            print("N =", self.N)
            print("MIN =", self.MIN)
            print("MAX =", self.MAX)
        return self.P, self.N, self.MIN, self.MAX

    def results(self, verbose=False):
        """Return and optionally print results

        Parameters
        ----------
        verbose : bool, optional
            Default False. If True, prints results to terminal

        Returns
        -------
        logZ : float
            estimated log-evidence
        dead_samples : list of Samples
            list of discarded Sample objects
        H : float
            estimated information
        count : int
            number of likelihood constraint iterations
        """
        if verbose:
            print("logZ =", self.logZ)
            print("H =", self.H)
            print("count =", self.count)
        return self.logZ, self.dead_samples, self.H, self.count

    def log_plus(self, x, y):
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


class Loglikelihood:
    """
    Log-likelihood object for Bayesian inference. This class must be subclassed
    to implement the evaluate function and define any custom behavior in the
    constructor. The default necessary parameters are listed below.

    Parameters
    ----------
    data : array-like
        array of data values
    """

    def __init__(self, data):
        """Class constructor"""
        self.data = data

    def evaluate(self, pvector):
        """Evaluate the log-likelihood function at pvector.

        Parameters
        ----------
        pvector : array-like
            Array of variates on the unit hypercube. These values will be
            transformed into model parameters.

        Returns
        -------
        logL : float
            evaluated log-likelihood
        """
        return 0.0


def importance_resampling(samples):
    """
    Resample weighted samples and return representative samples

    Parameters
    ----------
    samples : list
        A list of Sample instances. The logWt values are used to resample.

    Returns
    -------
    rep_samples : list
        A list of reweighted Sample instances, containing the same number of
        samples as the input
    """
    samples = deepcopy(samples)
    weighted_samples = sorted(samples, key=attrgetter('logWt'))
    max_log_w = samples[-1].logWt
    log_w = [sample.logWt - max_log_w for sample in weighted_samples]
    log_w = np.array(log_w, dtype=np.float64)
    w = np.exp(log_w)
    J = len(w)
    W = J * (w / w.sum())
    N = np.zeros((J, 1))
    u = random.rand(1)
    k = np.arange(0, J)[np.newaxis]
    S1 = np.hstack((0, W[:-1].cumsum()))[np.newaxis].T
    S2 = W.cumsum()[np.newaxis].T
    um = repmat(u, J, J)
    km = repmat(k, J, 1)
    S1m = repmat(S1, 1, J)
    S2m = repmat(S2, 1, J)
    Nm = ((um + km - S1m > np.zeros((J, J))).astype(np.int) -
          (um + km - S2m > np.zeros((J, J))).astype(np.int))
    N = Nm.sum(1)
    rep_samples = []
    for j in range(J):
        if N[j] > 0:
            sample = weighted_samples[j]
            sample.logWt = 0.0
            for i in range(N[j]):
                rep_samples.append(sample)
    return rep_samples
