#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:58:23
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-05 15:09:49

''' Main script to run the application. '''

import os
import psutil
from argparse import ArgumentParser
import numpy as np
import colorlover as cl

from WebSONIC import SONICViewer
from credentials import CREDENTIALS
import dash_auth

# If app is served via gunicorn -> use "production" settings
if psutil.Process(os.getppid()).name() == 'gunicorn':
    print('Serving via gunicorn')
    debug = False
    protect = True

# Otherwise -> determine settings by parsing command line arguments
else:
    ap = ArgumentParser()
    ap.add_argument('-d', '--debug', default=False, action='store_true', help='Run in Debug Mode')
    ap.add_argument('-o', '--opened', default=False, action='store_true',
                    help='Run without HTTP Authentification')
    args = ap.parse_args()
    debug = args.debug
    protect = not args.opened

# Set input parameters
inputs = {
    'diams': np.array([16.0, 32.0, 64.0]) * 1e-9,  # m
    'US_freqs': np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6]),  # Hz
    'US_amps': np.array([10, 20, 50, 100, 300, 600]) * 1e3,  # Pa
    'elec_amps': np.array([-25, -10, -5, -2, 2, 5, 10, 25]),  # mA/m2
    'PRFs': np.array([1e1, 1e2, 1e3, 1e4]),  # Hz
    'DCs': np.array([1., 5., 10., 25., 50., 75., 100.]),  # %
    'tstim': 250  # ms
}
ngraphs = 3

pltparams = {
    'tbounds': (-5., 300.),  # ms
    'colorset': [item for i, item in enumerate(cl.scales['8']['qual']['Set1']) if i != 5]
}

# Create app instance
app = SONICViewer(inputs, pltparams, ngraphs=ngraphs)
server = app.server
print('Created {}'.format(app))

# Protect app with/without HTTP authentification (activated by default)
if protect:
    dash_auth.BasicAuth(app, CREDENTIALS)
    print('Protected app with HTTP authentification')

if __name__ == '__main__':
    # Run app in standard mode (default, for production) or debug mode (for development)
    app.run_server(debug=debug)
