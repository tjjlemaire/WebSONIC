# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-29 16:15:17

''' Definition of application parameters. '''

import numpy as np
import matplotlib

# --------------------------------- Input parameters ---------------------------------

input_params = {
    'cell_type': {
        'label': 'Cell type',
        'values': ['RS', 'FS', 'LTS', 'IB', 'RE', 'TC', 'STN'],
        'default': 'STN'
    },
    'sonophore_radius': {
        'label': 'Sonophore radius',
        'unit': 'm',
        'values': np.logspace(np.log10(16.0), np.log10(64.0), 3) * 1e-9,
        'default': 32.0e-9  # m

    },
    'f_US': {
        'label': 'Frequency',
        'unit': 'Hz',
        'values': np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6]),
        'default': 500e3,
        'factor': 1e-3
    },
    'A_US': {
        'label': 'Amplitude',
        'unit': 'Pa',
        'values': np.array([10, 20, 40, 60, 80, 100, 300, 600]) * 1e3,
        'default': 20e3,
        'factor': 1e-3
    },
    'A_elec': {
        'label': 'Amplitude',
        'unit': 'mA/m2',
        'values': np.array([-25, -10, -5, -2, 2, 5, 10, 25]),
        'default': 10.
    },
    'tstim': {
        'label': 'Duration',
        'unit': 's',
        'values': np.array([20, 50, 100, 200, 500, 1000]) * 1e-3,
        'factor': 1e3,
        'default': 200e-3
    },
    'PRF': {
        'label': 'PRF',
        'unit': 'Hz',
        'values': np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3]),
        'default': 10.
    },
    'DC': {
        'label': 'Duty cycle',
        'unit': '%',
        'values': np.array([1., 5., 10., 25., 50., 75., 100.]),
        'default': 100.
    },
}


# --------------------------------- Plot parameters ---------------------------------

plt_params = {
    'colors': [matplotlib.colors.rgb2hex(c) for c in matplotlib.cm.get_cmap('tab10').colors]
}
