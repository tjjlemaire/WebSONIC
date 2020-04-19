# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-07 14:09:05
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-19 12:46:40
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-29 16:15:17

''' Definition of application parameters. '''

import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from PySONIC.utils import isWithin


class Parameter(metaclass=abc.ABCMeta):

    def __init__(self, label, default, disabled):
        self.label = label
        self.default = default
        self.disabled = disabled


class QualitativeParameter(Parameter):

    def __init__(self, label, values, default=None, disabled=False):
        self.values = values
        if default is None:
            default = values[0]
        super().__init__(label, default, disabled)


class QuantitativeParameter(Parameter):

    def __init__(self, label, default, unit, factor, disabled):
        super().__init__(label, default, disabled)
        self.unit = unit
        self.factor = factor

    @property
    @abc.abstractmethod
    def min(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max(self):
        raise NotImplementedError

    @property
    def amean(self):
        return (self.min + self.max) / 2

    @property
    def gmean(self):
        return np.sqrt(self.min * self.max)


class RangeParameter(QuantitativeParameter):

    def __init__(self, label, bounds, unit, factor=1., default=None, disabled=False,
                 scale='lin', n=100):
        self.bounds = bounds
        self.scale = scale
        self.n = n
        if default is None:
            default = self.gmean if self.scale == 'log' else self.amean
        else:
            default = isWithin(label, default, bounds)
        if scale == 'log':
            self.scaling_func = lambda x: np.power(10., x)
        else:
            self.scaling_func = lambda x: x
        super().__init__(label, default, unit, factor, disabled)

    @property
    def min(self):
        return self.bounds[0]

    @property
    def max(self):
        return self.bounds[1]


class SetParameter(QuantitativeParameter):

    def __init__(self, label, values, unit, factor=1., default=None, disabled=False):
        self.values = values
        if default is None:
            default = self.values[0]
        super().__init__(label, default, unit, factor, disabled)

    @property
    def min(self):
        return min(self.values)

    @property
    def max(self):
        return max(self.values)


ctrl_params = {
    'cell_type': QualitativeParameter(
        'Cell type', ['RS', 'FS', 'LTS', 'IB', 'RE', 'TC', 'STN'], default='RS'),
    'sonophore_radius': RangeParameter(
        'Sonophore radius', (16e-9, 64e-9), 'm', default=32e-9, scale='log', n=10),
    'sonophore_coverage_fraction': RangeParameter(
        'Coverage fraction', (1., 100.), '%', default=100., scale='lin', disabled=True, n=20),
    'f_US': RangeParameter(
        'Frequency', (20e3, 4e6), 'Hz', default=500e3, scale='log', n=20),
    'A_US': RangeParameter(
        'Amplitude', (10e3, 600e3), 'Pa', default=80e3, scale='log', n=100),
    'A_EL': RangeParameter(
        'Amplitude', (-25e-3, 25e-3), 'A/m2', factor=1e3, default=10e-3, n=100),
    'tstim': RangeParameter(
        'Duration', (20e-3, 1.0), 's', default=200e-3, scale='log', n=20),
    'PRF': RangeParameter(
        'PRF', (1e1, 1e3), 'Hz', default=2e1, scale='log', n=10),
    'DC': RangeParameter(
        'Duty cycle', (1., 100.), '%', default=100., scale='log', n=20)
}
