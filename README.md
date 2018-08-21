Description
============

Interactive web application to view the predicted electrical response of the different neuron types to ultrasonic stimuli for various combinations of sonication parameters, computed with the SONIC model. A comparative module to explore predicted responses of the same neurons to electrical stimuli (computed with standard Hodgkin-Huxley equations) is also included.

This application is built entirely in Python. It uses the Dash framework (https://dash.plot.ly/) for serving and client-side rendering, and the PySONIC package (https://c4science.ch/diffusion/4670/) for internal computations.

Installation
==================

This application is currently not implemented as a package, meaning that dependencies must be installed manually.

Open a terminal.

Activate a Python3 environment if needed:

	source <path_to_virtual_env>/bin activate

Check that the appropriate version of pip is activated:

	pip --version

Install all remote dependencies from pip:

	pip install dash==0.21.0  # the core dash backend
	pip install dash-renderer==0.11.3  # the dash front-end
	pip install dash-html-components==0.9.0  # HTML components
	pip install dash-core-components==0.21.0rc1  # Supercharged components (including tabs)
	pip install dash-auth==0.0.10  # the dash HTTP authentication module
	pip install plotly --upgrade  # the Plotly graphing library
	pip install colorlover==0.2.1  # graphing colors library
	pip install pysftp==0.2.9  # python wrapper for SFTP connections

Go to the PySONIC directory (where the setup.py file is located) and install it as a package:

	cd <path_to_directory>
	pip install -e .

The app should be ready to use.


Usage
=======

Local use
----------

To run the app locally, simply open a terminal at the location of the application directory and type in:

	python run.py

Then, visit [http://127.0.0.1:8050/viewer](http://127.0.0.1:8050/viewer) in your web browser. You should see the app displaying.


Remote deployment (Linux)
---------------------------

To deploy the application on a pre-configured linux machine, the best way to go is to use a Green Unicorn server.

Open a terminal.

Activate a Python3 environment if needed:

	source <path_to_virtual_env>/bin activate


Check that the appropriate version of pip is activated:

	pip --version

Install Green Unicorn as a python package

	pip install gunicorn

Then move to the application folder and serve the application with Gunicorn:

	gunicorn --bind 0.0.0.0:8050 run:server

Alternatively, you can serve the application in a separate, detached screen session:

	screen -d -S webnice_session -m gunicorn --bind 0.0.0.0:8050 run:server
