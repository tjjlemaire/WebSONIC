#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-24 17:15:57

from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(name='WebSONIC',
      version='1.0',
      description='Interactive web application to view the predicted electrical response of \
                   different neuron types to ultrasonic stimuli for various combinations of \
                   sonication parameters, computed with the **SONIC** model. A comparative module \
                   to explore predicted responses of the same neurons to electrical stimuli \
                   (computed with standard Hodgkin-Huxley equations) is also included.',
      long_description=readme(),
      url='???',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords=('SONIC NICE acoustic ultrasound ultrasonic neuromodulation neurostimulation excitation\
                 computational model intramembrane cavitation'),
      author='Théo Lemaire',
      author_email='theo.lemaire@epfl.ch',
      license='MIT',
      packages=['WebSONIC'],
      scripts=['run.py'],
      install_requires=[
          'dash==0.24.0',
          'dash-renderer==0.13.0',
          'dash-html-components==0.11.0'
          'dash-core-components==0.27.1',
          'dash-auth==1.0.0',
          'pysftp==0.2.9',
          'colorlover==0.2.1',
      ],
      zip_safe=False)
