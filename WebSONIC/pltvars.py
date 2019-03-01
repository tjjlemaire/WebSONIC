#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-01 19:22:38

''' Definition of neuron-specific plotting variables and their parameters. '''

import re


class PlotVariable:
    ''' Class representing a variable to plot. '''

    def __init__(self, names, desc, label=None, unit='-', factor=1, bounds=(-0.1, 1.1)):
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


# --------------------------------- Generic variables ---------------------------------

charge = PlotVariable('Qm', 'Membrane charge density', unit='nC/cm2', factor=1e5, bounds=(-90, 60))
potential = PlotVariable('Vm', 'Membrane potential', unit='mV', bounds=(-150, 60))


# --------------------------------- Other variables ---------------------------------

iH_reg_factor = PlotVariable('P0', 'iH regulating factor activation', label='iH reg.')

# Ca_conc = {'names': ['C_Ca'], 'desc': 'sumbmembrane Ca2+ concentration', 'label': '[Ca2+]',
#            'unit': 'uM', 'factor': 1e6, 'min': 0, 'max': 150.0}

# Leech specific variables
# iCa_gate = {'names': ['s'], 'desc': 'iCa gate opening', 'label': 'iCa gate', 'unit': '-',
#             'factor': 1, 'min': -0.1, 'max': 1.1}

# NaPump_reg = {'names': ['C_Na', 'A_Na'], 'desc': 'Sodium pump regulation',
#               'label': 'iNaPump reg.', 'unit': '-', 'factor': 1, 'min': -0.001, 'max': 0.01}

# iKCa_reg = {'names': ['C_Ca', 'A_Ca'], 'desc': 'Calcium-activated K+ current regulation',
#             'label': 'iKCa reg.', 'unit': '-', 'factor': 1, 'min': -1e-5, 'max': 1e-4}

# iKCa2_gate = {'names': ['c'], 'desc': 'iKCa gate opening', 'label': 'iKCa gate', 'unit': '-',
#               'factor': 1, 'min': -0.1, 'max': 1.1}

# NaPump2_reg = {'names': ['C_Na'], 'desc': 'Sodium pump regulation', 'label': '[Nai]',
#                'unit': 'mM', 'factor': 1e3, 'min': -1.0, 'max': 20.0}

# CaPump2_reg = {'names': ['C_Ca'], 'desc': 'Calcium pump regulation', 'label': '[Cai]',
#                'unit': 'uM', 'factor': 1e6, 'min': -0.1, 'max': 1.0}


# --------------------------------- Currents ---------------------------------

iNa = Current('iNa', 'Depolarizing Sodium current', gates=['m', 'h'])
iKd = Current('iKd', 'Delayed-recifier Potassium current', gates=['n'])
iM = Current('iM', 'Slow non-inactivating Potassium current', gates=['p'])
iCaT = Current('iCaT', 'Low-threshold (T-type) Calcium current', gates=['s', 'u'])
iCaTs = Current('iCaTs', 'Low-threshold (Ts-type) Calcium current', gates=['s', 'u'])
iH = Current('iH', 'Hyperpolarization-activated mixed cationic current',
             gates=['O', 'OL = 1 - O - C'])
iKleak = Current('iKleak', 'Leakage Potassium current')
iLeak = Current('iLeak', 'Leakage current')


# Neuron-specific variables dictionary
neuronvars = {
    'RS': {
        'desc': 'Cortical regular-spiking neuron',
        'currents': [iNa, iKd, iM, iLeak],
        'Vm0': -71.9  # mV
    },
    'FS': {
        'desc': 'Cortical fast-spiking neuron',
        'currents': [iNa, iKd, iM, iLeak],
        'Vm0': -71.4  # mV
    },
    'LTS': {
        'desc': 'Cortical low-threshold spiking neuron',
        'currents': [iNa, iKd, iM, iCaT, iLeak],
        'Vm0': -54.0  # mV
    },
    'RE': {
        'desc': 'Thalamic reticular neuron',
        'currents': [iNa, iKd, iCaTs, iLeak],
        'Vm0': -89.5  # mV
    },
    'TC': {
        'desc': 'Thalamo-cortical neuron',
        'currents': [iNa, iKd, iCaT, iH, iKleak, iLeak],
        'Vm0': -61.93  # mV
    }
    # 'LeechT': {
    #     'desc': 'Leech "touch" neuron',
    #     'vars_US': [charge, iNa_gates, iK_gate, iCa_gate, NaPump_reg, iKCa_reg],
    #     'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gate, NaPump_reg, iKCa_reg]
    # },
    # 'LeechP': {
    #     'desc': 'Leech "pressure" neuron',
    #     'vars_US': [charge, iNa_gates, iK_gate, iCa_gate, iKCa2_gate,
    #                 NaPump2_reg, CaPump2_reg],
    #     'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gate, iKCa2_gate, NaPump2_reg,
    #                   CaPump2_reg]
    # }
}




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
