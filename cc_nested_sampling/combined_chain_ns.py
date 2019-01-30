# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:21:01 2016

Combined-chain nested sampling
@author: wesley

Combined-chain nested sampling
Copyright (C) 2016  R. Wesley Henderson

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

from . import NS, mh
from multiprocessing import Pool
import numpy as np
from numpy.random import seed
from copy import copy


def worker(nsobj):
    """
    Worker function for multiprocessing.Pool

    Parameters
    ----------
    nsobj : NS
        A nested sampling object to start and collect results from

    Returns
    -------
    samples : list
        List of discarded samples from this NS object's run. Used to compute
        evidence later.
    """
    seed()
    output = nsobj.run()
    samples = output[1]
    return samples


class ParallelNS(NS):
    """
    Initialize the NS nested sampling object.

    Parameters
    ----------
    P : int
        number of model parameters to pass to log-likelihood function
    M : int
        number of paralle chains to run with N live samples
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
    def __init__(self, P, M, lhood, explore=mh, N=20, MIN=1, MAX=1e6,
                 verbose=False, *args, **kwargs):
        self.P = P
        self.M = M
        self.lhood = lhood
        self.explore = explore
        self.N = N
        self.MIN = MIN
        self.MAX = MAX
        self.verbose = verbose

    def run(self):
        """Run nested sampling algorithm. If M > 1, run in parallel.

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
        if self.M < 1:
            raise ValueError("M must be an integer >= 1")
        elif self.M == 1:
            return self.run_serial()
        else:
            return self.run_parallel()

    def run_parallel(self):
        """Run parallel nested sampling algorithm

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
        nsobjs = []
        for i in range(self.M):
            nsobjs.append(NS(self.P, self.lhood, self.explore, self.N,
                             self.MIN, self.MAX, self.verbose))
        sample_list_list = None
        with Pool() as pool:
            sample_list_list = pool.map(worker, nsobjs)
        samples = [sample for sample_list in sample_list_list for sample in
                   sample_list]
        # Sort the combined sample list and recompute the evidence and
        # information
        samples = sorted(samples)
        MN = self.M * self.N
        log_width = np.log(1 - np.exp(-1/MN))
        logZ = -np.inf
        H = 0.0
        for sample in samples:
            sample.logWt = sample.logL + log_width
            logZnew = self.log_plus(logZ, sample.logWt)
            H_new = (np.exp(sample.logWt - logZnew)*sample.logL +
                     np.exp(logZ - logZnew)*(H + logZ) - logZnew)
            if ~np.isnan(H_new):
                H = copy(H_new)
            logZ = copy(logZnew)
            log_width -= 1/MN
        self.logZ = logZ
        self.H = H
        self.dead_samples = samples
        self.count = len(samples)
        return self.logZ, self.dead_samples, self.H, self.count
