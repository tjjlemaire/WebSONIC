# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-10 16:13:58

''' Definition of applicaiton parameters. '''

import numpy as np
import colorlover as cl

# Input parameters
inputparams = dict(
    diams=np.array([16.0, 32.0, 64.0]) * 1e-9,  # m
    US_freqs=np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6]),  # Hz
    US_amps=np.array([10, 20, 40, 60, 80, 100, 300, 600]) * 1e3,  # Pa
    elec_amps=np.array([-25, -10, -5, -2, 2, 5, 10, 25]),  # mA/m2
    PRFs=np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3]),  # Hz
    DCs=np.array([1., 5., 10., 25., 50., 75., 100.]),  # %
    tstim=250  # ms
)

inputdefaults = dict(
    diams=32.0e-9,  # m
    US_freqs=500e3,  # Hz
    US_amps=80e3,  # Pa
    elec_amps=10,  # mA/m2
    PRFs=1e2,  # Hz
    DCs=100.  # %
)

# Plotting parameters
pltparams = dict(
    tbounds=(-5., 300.),  # ms
    colorset=[item for i, item in enumerate(cl.scales['8']['qual']['Set1']) if i != 5]
)

# Number of graphs
ngraphs = 3
