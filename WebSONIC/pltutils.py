#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 16:24:53

''' Definition of neuron-specific plotting variables and their parameters. '''

import re
import numpy as np


class PlotVariable:
    ''' Class representing a variable to plot. '''

    def __init__(self, names, desc, label=None, unit='-', factor=1, bounds=(-0.1, 1.1), y0=None):
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

    def getData(self, df, nonset=2):
        data = []
        for n in self.names:

            # extract signal from dataframe
            if '=' in n:
                y = eval(extractFromFormula(n, wrapleft='df["', wrapright='"]')[1]).values
            else:
                y = df[n].values

            # add onset
            y0 = self.y0 if self.y0 is not None else y[0]
            y = np.hstack((np.array([y0] * nonset), y))

            # rescale with appropriate factor
            y *= self.factor

            # add to signals list
            data.append(y)

        return data


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
                label='{} gates'.format(self.name))
        else:
            return None


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
                'Qm', 'Membrane charge density', unit='nC/cm2', factor=1e5, bounds=(-90, 60)),
            PlotVariable(
                'Vm', 'Membrane potential', unit='mV', bounds=(-150, 60), y0=Vm0)
        ]
        for c in self.currents:
            if c.gates is not None:
                self.pltvars.append(c.gatingVariables())
            if c.internals is not None:
                self.pltvars += c.internals


def extractFromFormula(exp, wrapleft='', wrapright=''):
    ''' Extract a variable name and the necessary string to avaluate its value,
        from an algebraic formula.

        :param exp: algebraic formula (string)
        :param wrapleft: string to be placed on the left side of output variables
        :param wrapleft: string to be placed on the right side of output variables
        :return: 2-tuple with the name of the output variable, and the expression to evaluate it.
    '''

    var_pattern = '[A-Za-z][A-Za-z0-9_]*'
    equality_pattern = '({}) = (.*)'.format(var_pattern)

    # Extract output name and output expression from formula
    mo = re.fullmatch(equality_pattern, exp)
    outname = mo.group(1)
    outexp = mo.group(2)

    # Wrap container around all variables in the output expression
    outexp = re.sub(
        '({})'.format(var_pattern),
        r'{}\1{}'.format(wrapleft, wrapright),
        outexp
    )

    return outname, outexp
