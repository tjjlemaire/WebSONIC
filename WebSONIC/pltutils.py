#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-06 17:08:48

''' Definition of neuron-specific plotting variables and their parameters. '''

import re
import numpy as np
import colorlover as cl

set1 = cl.scales['8']['qual']['Set1']
set2 = cl.scales['8']['qual']['Dark2']
var_pattern = '[A-Za-z][A-Za-z0-9_]*'
func_pattern = r'({})(\(.*\))'.format(var_pattern)


class PlotVariable:
    ''' Class representing a variable to plot. '''

    aliases = {
        'OL': '1 - O - C'
    }

    def __init__(self, names, desc, label=None, unit='-', factor=1,
                 bounds=None, y0=None, colors=None):
        if isinstance(names, str):
            names = [names]
        self.names = names
        self.desc = desc
        if label is not None:
            self.label = label
        else:
            if len(names) == 1:
                self.label = names[0]
            else:
                self.label = self.desc
        self.unit = unit
        self.factor = factor
        self.bounds = bounds
        self.y0 = y0
        if colors is not None:
            self.colors = colors
        else:
            self.colors = set1[:len(self.names)]

    def getData(self, obj, df, nonset=2):
        data = []
        names = []
        for n in self.names:
            # extract signal
            s1, s2 = extract(n, df, obj, self.aliases)
            names.append(s1)
            y = eval(s2).values

            # add onset
            y0 = self.y0 if self.y0 is not None else y[0]
            y = np.hstack((np.array([y0] * nonset), y))

            # rescale with appropriate factor
            y *= self.factor

            # add to signals list
            data.append(y)

        return names, np.array(data)


class Current:
    ''' Class representing a membrane current object with specific gating kinetics. '''

    def __init__(self, name, desc, gates=None, internals=None):
        self.name = name
        self.desc = desc
        if isinstance(gates, str):
            gates = [gates]
        self.gates = gates
        self.internals = internals

    def gatingVariables(self):
        ''' Return a list of variables that can be plotted for a given current. '''
        if self.gates is not None:
            return PlotVariable(
                self.gates, '{} ({}) kinetics'.format(self.desc, self.name),
                label='{} gates'.format(self.name), bounds=(-0.1, 1.1))
        else:
            return None

    def expression(self):
        return '{}({}Vm)'.format(self.name, '{}, '.format(
            ', '.join(self.gates)) if self.gates is not None else '')


class CellType:
    ''' Class representing a cell type with specific membrane currents and resting potential. '''

    def __init__(self, name, desc, currents, Vm0=0.0):
        self.name = name
        self.desc = desc
        self.currents = currents
        self.Vm0 = Vm0  # mV

        # Populate list of variables that can be plotted for the cell type
        self.pltvars = [
            PlotVariable(
                'Qm', 'Membrane charge density', unit='nC/cm2', factor=1e5, bounds=(-90, 60),
                colors=['black']),
            PlotVariable(
                'Vm', 'Membrane potential', unit='mV', bounds=(-150, 60), y0=Vm0,
                colors=['darkgray'])
        ]
        self.pltvars.append(PlotVariable(
            [i.expression() for i in self.currents],
            'Membrane currents', label='I', unit='A/m2', factor=1e-3, bounds=(-10, 10),
            colors=set2[:len(self.currents)]
        ))
        for i in self.currents:
            if i.gates is not None:
                self.pltvars.append(i.gatingVariables())
            if i.internals is not None:
                self.pltvars += i.internals


def wrap(exp, wrapleft='', wrapright=''):
    return re.sub(
        '({})'.format(var_pattern),
        r'{}\1{}'.format(wrapleft, wrapright),
        exp)


def extract(varname, df, obj, aliases):
    if varname in df:
        return varname, 'df["{}"]'.format(varname)
    elif varname in aliases:
        varexp = aliases[varname]
        return varname, wrap(varexp, wrapleft='df["', wrapright='"]')
    else:
        mo = re.fullmatch(func_pattern, varname)
        if mo is not None:
            funcname = mo.group(1)
            funcexp = mo.group(2)
            funcexp = re.sub(
                '({})'.format(var_pattern),
                lambda mo: extract(mo.group(1), df, obj, aliases)[1],
                funcexp)
            return funcname, 'obj.{}{}'.format(funcname, funcexp)
        else:
            raise KeyError('Could not extract {}'.format(varname))
