#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:41:07
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-09 18:56:33

''' Create a Flask server instance '''
import os
import glob
from flask import Flask, send_from_directory

# Create server instance
server = Flask(__name__)

image_directory = 'img/'

# Add specific route to serve static files
static_route = '/static/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.gif'.format(image_directory))]

css_route = '/css/'
css_directory = os.getcwd()
stylesheets = ['stylesheet.css']


@server.route('{}<image_path>'.format(static_route))
def serve_image(image_path):
    image_name = '{}_neuron_anim.gif'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return send_from_directory(image_directory, image_name)


@server.route('{}<stylesheet>'.format(css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception('"{}" is excluded from the allowed static files'.format(stylesheet))
    return send_from_directory(css_directory, stylesheet)
