#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:41:07
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-02-26 18:13:15

''' Create a Flask server instance '''
import os
import glob
from flask import Flask, send_from_directory

# Create server instance
server = Flask(__name__)

image_directory = 'img/'

# Add specific route to serve static files
static_route = '/static/'
extensions = ['png', 'svg']
list_of_images = []
for ext in extensions:
    list_of_images += [os.path.basename(x) for x in glob.glob('{}*.{}'.format(image_directory, ext))]

css_route = '/css/'
css_directory = os.getcwd()
stylesheets = ['dash_styles.css', 'my_styles.css']

# Serve local static image files
# e.g. <IP_adress>:<port>/static/RS_mech.png
@server.route('{}<image_path>'.format(static_route))
def serve_image(image_path):
    image_name = image_path
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return send_from_directory(image_directory, image_name)


@server.route('{}<stylesheet>'.format(css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception('"{}" is excluded from the allowed static files'.format(stylesheet))
    return send_from_directory(css_directory, stylesheet)


@server.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(server.root_path, 'img'), 'nbls.svg',
                               mimetype='image/vnd.microsoft.icon')
