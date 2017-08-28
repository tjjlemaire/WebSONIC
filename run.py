#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-11 18:58:23
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 15:57:53

''' Run the application on the flask server. '''

from server import server
from viewer import app as viewer


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8050)
