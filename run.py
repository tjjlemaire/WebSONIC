#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:58:23
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-18 16:26:57

''' Main script to run the application. '''

import os
import psutil
from argparse import ArgumentParser

from WebSONIC import SONICViewer
from WebSONIC.params import input_params, plt_params
from credentials import CREDENTIALS
import dash_auth

# Determine if app is served via gunicorn or normally ("basic" flask serving)
is_gunicorn = psutil.Process(os.getppid()).name() == 'gunicorn'

# If app is served via gunicorn -> use "production" settings
if is_gunicorn:
    print('Serving via gunicorn')
    debug = False
    protect = True
    ngraphs = 3

# Otherwise -> determine settings by parsing command line arguments
else:
    ap = ArgumentParser()
    ap.add_argument('-d', '--debug', default=False, action='store_true', help='Run in Debug Mode')
    ap.add_argument('-o', '--opened', default=False, action='store_true',
                    help='Run without HTTP Authentification')
    ap.add_argument('-n', '--ngraphs', type=int, default=3, help='Number of parallel graphs')
    args = ap.parse_args()
    debug = args.debug
    protect = not args.opened
    ngraphs = args.ngraphs

# Create app instance
app = SONICViewer(input_params, plt_params, ngraphs)
app.scripts.config.serve_locally = True
print('Created {}'.format(app))

# Add underlying server instance to module global scope (for gunicorn use)
if is_gunicorn:
    server = app.server

# Protect app with/without HTTP authentification (activated by default)
if protect:
    dash_auth.BasicAuth(app, CREDENTIALS)
    print('Protected app with HTTP authentification')

if __name__ == '__main__':
    # Run app in standard mode (default, for production) or debug mode (for development)
    app.run_server(debug=debug)
