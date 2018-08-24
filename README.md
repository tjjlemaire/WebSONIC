Description
============

Interactive web application to view the predicted electrical response of different neuron types to ultrasonic stimuli for various combinations of sonication parameters, computed with the SONIC model. A comparative module to explore predicted responses of the same neurons to electrical stimuli (computed with standard Hodgkin-Huxley equations) is also included.

This application is built entirely in Python. It uses the Dash framework (https://dash.plot.ly/) for serving and client-side rendering, and the PySONIC package (https://c4science.ch/diffusion/4670/) for internal computations.

Installation
==================

From a terminal, activate a Python3 environment if needed:

	source <path_to_virtual_env>/bin activate

Check that the appropriate version of pip is activated:

	pip --version

Install the PySONIC dependency locally:

	cd <path_to_PySONIC_dir/setup.py>
	pip install -e .

Install the WebSONIC package :

	cd <path_to_PySONIC_dir/setup.py>
	pip install -e .


All remote dependencies will be automatically installed.

The app should be ready to use.

Usage
=======

Local use
----------

You can run the application from a local terminal with a single command line (in the app directory):

	python run.py [-d]

The [-d] or [--debug] extra parameter is to run the application in "debug" mode (restarting upon file save).

Then, open a browser at [http://127.0.0.1:8050/viewer](http://127.0.0.1:8050/viewer) to use the application.


Remote deployment (Linux)
---------------------------

To deploy the application on a pre-configured linux machine, the best way to go is to use a Green Unicorn server.

From a terminal, activate a Python3 environment if needed:

	source <path_to_virtual_env>/bin activate

Check that the appropriate version of pip is activated:

	pip --version

Install Green Unicorn as a python package

	pip install gunicorn

Then move to the application folder and serve the application with Gunicorn:

	gunicorn --bind 0.0.0.0:8050 run:server

Alternatively, you can serve the application in a separate, detached screen session:

	screen -d -S webnice_session -m gunicorn --bind 0.0.0.0:8050 run:server
