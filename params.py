# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-07 14:09:05
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-22 11:58:58
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-29 16:15:17

''' Definition of control parameters. '''

import abc
import numpy as np
from PySONIC.utils import isWithin, friendlyLogspace


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
                 scale='lin', n=100, bases=None, round_factor=None):
        if bases is None:
            bases = range(1, 10)
        self.bounds = np.asarray(bounds)
        self.scale = scale
        self.n = n
        self.bases = bases
        if scale == 'lin':
            self.values = np.linspace(*self.bounds, self.n)
        elif scale == 'log':
            self.values = np.logspace(*np.log10(self.bounds), self.n)
        elif scale == 'friendly-log':
            self.values = friendlyLogspace(*self.bounds, bases=self.bases)
            self.n = self.values.size
        if round_factor is not None:
            self.values = np.round(self.values * round_factor) / round_factor
        if default is None:
            default = self.gmean if self.scale == 'log' else self.amean
        else:
            default = isWithin(label, default, bounds)
        self.idefault = np.argmin(np.abs(self.values - default))
        default = self.values[self.idefault]
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
