# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-11 18:58:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-22 12:13:11

''' Main script to run the application. '''

import os
import psutil
from argparse import ArgumentParser

from viewer import SONICViewer

# Determine if app is served via gunicorn or normally ("basic" flask serving)
is_gunicorn = psutil.Process(os.getppid()).name() == 'gunicorn'
if is_gunicorn:
    print('Serving via gunicorn')
    debug = False
    verbose = False

else:
    # Determine settings by parsing command line arguments
    ap = ArgumentParser()
    ap.add_argument(
        '-d', '--debug', default=False, action='store_true', help='Run in Debug Mode')
    ap.add_argument(
        '-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    args = ap.parse_args()
    debug = args.debug
    verbose = args.verbose

# Create app instance
app = SONICViewer(verbose=verbose)

# Add underlying server instance to module global scope (for gunicorn use)
if is_gunicorn:
    server = app.server

if __name__ == '__main__':
    # Run app in standard mode (default, for production) or debug mode (for development)
    app.run_server(debug=debug)
