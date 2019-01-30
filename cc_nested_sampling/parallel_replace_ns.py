# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:49:52 2016

Nested sampling with multiple discard-and-replace
@author: wesley

Parallel discard-and-replace nested sampling
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

from . import NS, mh, Sample
from multiprocessing import Pool
import numpy as np
from copy import deepcopy


class PReplaceNS(NS):
    """
    Initialize the NS nested sampling object.

    Parameters
    ----------
    P : int
        number of model parameters to pass to log-likelihood function
    R : int
        number of samples to discard and replace at once
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
    def __init__(self, P, R, lhood, explore=mh, N=20, MIN=1, MAX=1e6,
                 verbose=False, *args, **kwargs):
        self.P = P
        self.R = R
        self.lhood = lhood
        self.explore = explore
        self.N = N
        self.MIN = MIN
        self.MAX = MAX
        self.verbose = verbose

    def run(self):
        """Run nested sampling algorithm. If R > 1, do parallel discard and
        replace.

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
        if self.R < 1:
            raise ValueError("M must be an integer >= 1")
        elif self.R == 1:
            return self.run_serial()
        else:
            return self.run_parallel()

    def run_parallel(self):
        """Run nested sampling algorithm with parallel discard and replace

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
        Elogt = -1 * self.R / self.N
        self.logZ = -np.inf
        self.H = np.float64(0.0)
        log_width = np.log(1-np.exp(Elogt))
        live_samples = [Sample(self.P, self.lhood) for i in range(self.N)]
        self.dead_samples = []
        iterate = True
        self.count = 0
        with Pool() as p:
            while iterate:
                # min_sample = min(live_samples, key=attrgetter('logL'))
                live_samples = sorted(live_samples)
                logL_min = live_samples[self.R-1].logL
                if self.verbose:
                    print("P =", self.P, " count =", self.count, " L =",
                          logL_min)
                newlogWt = logL_min + log_width
                live_samples[self.R-1].logWt = newlogWt
                logZnew = self.log_plus(self.logZ, newlogWt)
                H_new = (np.exp(newlogWt - logZnew)*logL_min +
                         np.exp(self.logZ - logZnew)*(self.H + self.logZ) -
                         logZnew)
                if ~np.isnan(H_new):
                    self.H = H_new
                self.logZ = logZnew
                min_samples = deepcopy(live_samples[:self.R])
                new_samples = p.starmap(self.explore,
                                        [(live_samples[self.R:], logL_min,
                                          self.R-1, self.lhood) for i in
                                         range(self.R)])
                live_samples[:self.R] = deepcopy(new_samples)
                self.dead_samples.append(deepcopy(min_samples))
                log_width += Elogt
                self.count += 1
                if self.count > self.MIN and self.count > (2 * self.H *
                                                           self.N / self.R):
                    iterate = False
                elif self.count >= self.MAX:
                    iterate = False
        if self.verbose:
            print("Done!")
        return self.logZ, self.dead_samples, self.H, self.count
