# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:15
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-24 19:36:57
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-29 13:34:25

from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(
    name='WebSONIC',
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
        'dash>=0.39.0',
        'dash-daq>=0.1.4',
        'dash-bootstrap-components>=0.6.3',
        'plotly>=3.7.1',
        'psutil>=5.4'
    ],
    zip_safe=False
)
