General instructions on how to deploy the webNICE app on a pre-configured linux machine.

Activate Python3 virtual environment:
$ source /path/to/virtualenv/bin/activate

Move to application folder:
$ cd /path/to/application

Serve the application with Gunicorn:
$ gunicorn --bind 0.0.0.0:8050 run:server

Or in a separate, detached screen session:
$ screen -d -S webnice_session -m gunicorn --bind 0.0.0.0:8050 run:server