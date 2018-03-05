Description
============

**WebNICE** is an interactive web viewer built entirely in Python using the Dash framework that allows to compare the predicted responses of different types of neurons to both electrical and ultrasonic stimuli.
These responses are computed using two types of point-neuron models:
- ***Hodgkin-Huxley*** models for electrical stimulation
- variants of a new, effective implementation of the ***Neuronal Intramembrane Cavitation Excitation (NICE)*** model by M. Plaksin, E. Kimmel and S. Shoham for ultrasonic stimulation</li>

For a given neuron and modality, the interface allows to explore the stimulation parameter space in two ways: by moving sliders and loading precomputed results or by manually entering custom parameters and running point-neuron simulations on-the-fly using the **PointNICE** package.


Installation
==================

WebNICE is currently not implemented as a package, meaning that dependencies must be installed manually.

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

Go to the PointNICE directory (where the setup.py file is located) and install it as a package:

	cd <path_to_directory>
	pip install -e .

The WebNICE app should be ready to use.


Usage
=======

Local use
----------

To run the app locally, simply open a terminal at the location of the WebNICE directory and type in:

	python run.py

Then, visit [http://127.0.0.1:8050/viewer](http://127.0.0.1:8050/viewer) in your web browser. You should see the app displaying.


Remote deployment (Linux)
---------------------------

To deploy the WebNICE app on a pre-configured linux machine, the best way to go is to use a Green Unicorn server.

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
