# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:15
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-24 19:36:42
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:58:23
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-30 11:59:12

''' Main script to run the application. '''

import os
import psutil
from argparse import ArgumentParser

from WebSONIC import SONICViewer
from WebSONIC.params import input_params, plt_params

# Determine if app is served via gunicorn or normally ("basic" flask serving)
is_gunicorn = psutil.Process(os.getppid()).name() == 'gunicorn'
if is_gunicorn:
    print('Serving via gunicorn')
    ngraphs = 3
    debug = False
    testUI = False
    auth = False
    verbose = False

else:
    # Determine settings by parsing command line arguments
    ap = ArgumentParser()
    ap.add_argument('-d', '--debug', default=False, action='store_true', help='Run in Debug Mode')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-n', '--ngraphs', type=int, default=3, help='Number of parallel graphs')
    ap.add_argument('-t', '--testUI', default=False, action='store_true', help='Test UI only')
    args = ap.parse_args()
    ngraphs = args.ngraphs
    debug = args.debug
    testUI = args.testUI
    auth = args.auth
    verbose = args.verbose

# Create app instance
app = SONICViewer(input_params, plt_params, ngraphs, no_run=testUI, verbose=verbose)
print('Created {}'.format(app))

# Add underlying server instance to module global scope (for gunicorn use)
if is_gunicorn:
    server = app.server

if __name__ == '__main__':
    # Run app in standard mode (default, for production) or debug mode (for development)
    app.run_server(debug=debug)
