List of packages to install from pip:
- dash (0.21.0): the core dash backend
- dash-renderer (0.11.3): the dash front-end
- dash-html-components (0.9.0): HTML components
- dash-core-components (0.21.0rc1): # Supercharged components (including tabs)
- dash-auth (0.0.10): the dash HTTP authentication module
- plotly (upgrade): the Plotly graphing library
- colorlover (0.2.1): graphing colors library
- pysftp (0.2.9): python wrapper for SFTP connections
- PointNICE (local install)

General instructions on how to deploy the webNICE app on a pre-configured linux machine.

Activate Python3 virtual environment:
$ source /path/to/virtualenv/bin/activate

Move to application folder:
$ cd /path/to/application

Serve the application with Gunicorn:
$ gunicorn --bind 0.0.0.0:8050 run:server

Or in a separate, detached screen session:
$ screen -d -S webnice_session -m gunicorn --bind 0.0.0.0:8050 run:server