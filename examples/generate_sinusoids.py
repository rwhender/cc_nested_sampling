# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:12:36 2014

@author: Wesley

Script for generating sinusoid likelihood simulated data
Copyright (C) 2014  R. Wesley Henderson

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

import dill
import numpy as np
from scipy.stats import norm
from copy import copy

if __name__ == "__main__":
    fs = 64
    time = np.random.rand(64)
    time.sort()
    As = np.array([-0.9241, -0.1988])
    Bs = np.array([-0.3822, 0.9800])
    freqs = np.array([2.0, 5.437])
    n = len(time)
    signal = np.zeros(n)
    for A, B, f in zip(As, Bs, freqs):
        signal = (signal + A * np.cos(2 * np.pi * f * time) +
                  B * np.sin(2 * np.pi * f * time))
    # Generate noise
    sigma = 0.5
    noise = norm.rvs(loc=0, scale=sigma, size=n)
    old_signal = copy(signal)
    signal += noise
    signal = signal[np.newaxis].T
    time = time[np.newaxis].T
    with open('sinusoid_data.dill', 'wb') as f:
        dill.dump(signal, f, dill.HIGHEST_PROTOCOL)
        dill.dump(time, f, dill.HIGHEST_PROTOCOL)
        dill.dump(noise, f, dill.HIGHEST_PROTOCOL)
        dill.dump(fs, f, dill.HIGHEST_PROTOCOL)
        dill.dump(As, f, dill.HIGHEST_PROTOCOL)
        dill.dump(Bs, f, dill.HIGHEST_PROTOCOL)
        dill.dump(freqs, f, dill.HIGHEST_PROTOCOL)
