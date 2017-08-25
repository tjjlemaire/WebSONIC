General instructions on how to deploy the webNICE app on a pre-configured linux machine, and make it publicly available on the EPFL domain.

Activate Python3 virtual environment:
$ source ~/venvs/flaskproj/bin/activate

Move to application folder:
$ cd ~/Documents/tlemaire/webNICE

Serve the application with Gunicorn:
$ gunicorn --bind 0.0.0.0:8050 run:server

The webNICE app is now available at:
128.178.51.155:8050/viewer
username: tne2017
password: nice