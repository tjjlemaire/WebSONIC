#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-10 14:55:11


''' Open SFTP channel and set the root of the remote DATA directory. '''

import base64
import pysftp

data_root = 'WebNICE_data'

# server information
host_b64 = b'MTI4LjE3OC44NC45Mg=='  # IP address of data server (base-64 encoded)
user_b64 = b'dG5ld2ViYXBw'  # username (base-64 encoded)
passwd_b64 = b'bWV5cmluOTI='  # password (base-64 encoded)

# no key (trusted communication line)
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

# opening sftp channel (but no closing it!)
channel = pysftp.Connection(host=base64.b64decode(host_b64).decode('utf-8'),
                            port=442,
                            username=base64.b64decode(user_b64).decode('utf-8'),
                            password=base64.b64decode(passwd_b64).decode('utf-8'),
                            cnopts=cnopts)
