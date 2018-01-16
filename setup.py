#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-16 20:36:33

from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(name='WebNICE',
      version='1.0',
      description='An interactive web viewer built entirely in Python using the Dash framework\
                   that allows to compare the predicted responses of different types of neurons\
                   to both electrical and ultrasonic stimuli.',
      long_description=readme(),
      url='https://tnewebnice.epfl.ch',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords=('ultrasound ultrasonic neuromodulation neurostimulation excitation\
                 biophysical model intramembrane cavitation NICE'),
      author='Théo Lemaire',
      author_email='theo.lemaire@epfl.ch',
      license='MIT',
      packages=['WebNICE'],
      scripts=['run.py'],
      install_requires=[
          'dash>=0.19.0',
          'dash-html-components>=0.8.0',
          'dash-core-components==0.15.0rc1',
          'dash-renderer>=0.11.2',
          'dash-auth>=0.0.10'
          'plotly>=2.2.3',
          'numpy>=1.10',
          'pandas>=0.20.3',
          'colorlover>=0.2.1',
          'pysftp>=0.2.9'
      ],
      zip_safe=False)
