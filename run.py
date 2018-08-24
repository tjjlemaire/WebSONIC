#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:58:23
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-24 17:39:13

''' Run the application on the flask server. '''

import os
import numpy as np
import colorlover as cl
from flask import Flask
from enum import Enum

from WebSONIC import SONICViewer, connectSSH
from credentials import CREDENTIALS


class RunMode(Enum):
    dev = 1
    prod = 2


# Create server instance
server = Flask('server')

# Set up SSH channel
channel = connectSSH()
remoteroot = 'WebNICE_data'
tmpdir = os.path.join(os.getcwd(), 'tmp')

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

runmode = RunMode.prod

if __name__ == '__main__':

    # Create app instance
    app = SONICViewer(server, tmpdir, remoteroot, channel, inputs, pltparams,
                      ngraphs=ngraphs, credentials=CREDENTIALS)
    print('Started {}'.format(app))

    # Run web app
    if runmode == RunMode.dev:
        app.run_server(debug=True)  # run in debug mode (restarts upon every file save)
    elif runmode == RunMode.prod:
        server.run(host='0.0.0.0', port=8050)  # run in standard mode (for production)
    else:
        print('Error: unkown run mode')
