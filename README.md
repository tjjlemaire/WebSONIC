# Description

`WebSONIC` is an interactive web application allowing to visualize the predicted electrical response of different neuron types to ultrasonic stimuli for various combinations of sonication parameters, computed with the **SONIC** model [1]. A comparative module to explore predicted responses of the same neurons to electrical stimuli (computed with standard Hodgkin-Huxley equations) is also included.

## Content of repository

- the `viewer` module contains a core `SONICViewer` class inherited from `dash.Dash` that defines the layout and all the callbacks of the web app.
- the `components` module defines custom UI components used in the layout.
- the `params` module defines input and plotting parameters used by the web app.
- the **assets** folder contains CSS style sheets and static files used by the web app.
- the `index.html` defines a default home page used if the web app is hosted on a server.
- the `run.py` script allows to launch the app from the commandmline with different options.

## Dependencies

This application is built in Python. It uses the **Dash** framework (https://dash.plot.ly/) for serving and client-side rendering, and the **NEURON** simulation environment (https://www.neuron.yale.edu/neuron/) to run simulations. It depends on two other Python papckages:
- `PySONIC` (https://c4science.ch/diffusion/4670/) defines the **SONIC model** and provides utilities
- `ExSONIC` (https://c4science.ch/diffusion/7145/) handles the communication with **NEURON**

# Installation

- From a terminal, activate a Python3 environment if needed:

`source <path_to_virtual_env>/bin activate`

- Check that the appropriate version of pip is activated:

`pip --version`

- Install local dependencies (PySONIC and ExSONIC packages):

```
cd <path_to_PySONIC_dir/setup.py>
pip install -e .
cd <path_to_ExSONIC_dir/setup.py>
pip install -e .
```

- Install remote dependencies (dash and other packages):

```
cd <path_to_app_dir>
pip install -r requirements.txt
```

That's it!

# Usage

## Local use

You can run the application from a local terminal with a single command line (in the app directory):

`python run.py`

The following command line arguments are available:
- `-d` / `--debug`: run the application in **debug** mode (restarting upon file save).
- `-v` / `--verbose`: add verbosity
- `-n` / `--ngraphs`: specify the number of graphs in the web app
- `-t` / `--testUI`: test the we app UI, without running internal simulations

Then, open a browser at [http://127.0.0.1:8050/viewer](http://127.0.0.1:8050/viewer) to use the application.

## Remote deployment (Linux)

To deploy the application on a pre-configured linux machine, the best way to go is to use a Green Unicorn server:

- From a terminal, activate a Python3 environment if needed:

`source <path_to_virtual_env>/bin activate`

- Check that the appropriate version of pip is activated:

`pip --version`

- Install Green Unicorn as a python package

`pip install gunicorn`

- Move to the application folder and serve the application with Gunicorn:

`gunicorn --bind 0.0.0.0:8050 run:server`

- Alternatively, you can serve the application in a separate, detached screen session:

`screen -d -S webnice_session -m gunicorn --bind 0.0.0.0:8050 run:server`

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.

