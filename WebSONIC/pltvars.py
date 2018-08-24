#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-23 09:20:33

''' Definition of neuron-specific plotting variables and their parameters. '''

charge = {'names': ['Qm'], 'desc': 'charge density', 'label': 'charge', 'unit': 'nC/cm2',
          'factor': 1e5, 'min': -90, 'max': 60}

potential = {'names': ['Vm'], 'desc': 'membrane potential', 'label': 'potential', 'unit': 'mV',
             'factor': 1e0, 'min': -150, 'max': 60}

deflection = {'names': ['Z'], 'desc': 'leaflets deflection', 'label': 'deflection', 'unit': 'nm',
              'factor': 1e9, 'min': -0.1, 'max': 0.5}

gas = {'names': ['ng'], 'desc': 'gas content', 'label': 'gas', 'unit': '1e-22 mol',
       'factor': 1e22, 'min': 1.5, 'max': 2.0}

iNa_gates = {'names': ['m', 'h'], 'desc': 'iNa gates opening', 'label': 'iNa gates', 'unit': '-',
             'factor': 1, 'min': -0.1, 'max': 1.1}

iK_gate = {'names': ['n'], 'desc': 'iK gate opening', 'label': 'iK gate', 'unit': '-',
           'factor': 1, 'min': -0.1, 'max': 1.1}

iM_gate = {'names': ['p'], 'desc': 'iM gate opening', 'label': 'iM gate', 'unit': '-',
           'factor': 1, 'min': -0.1, 'max': 1.1}

iCa_gates = {'names': ['s', 'u'], 'desc': 'iCa gates opening', 'label': 'iCa gates', 'unit': '-',
             'factor': 1, 'min': -0.1, 'max': 1.1}

# iH_gates = {'names': ['O', 'OL'], 'desc': 'iH gates opening', 'label': 'iH gates', 'unit': '-',
#             'factor': 1, 'min': -0.1, 'max': 1.1}

iH_reg_factor = {'names': ['P0'], 'desc': 'iH regulating factor activation',
                 'label': 'iH reg.', 'unit': '-', 'factor': 1, 'min': -0.1, 'max': 1.1}

Ca_conc = {'names': ['C_Ca'], 'desc': 'sumbmembrane Ca2+ concentration', 'label': '[Ca2+]',
           'unit': 'uM', 'factor': 1e6, 'min': 0, 'max': 150.0}

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


# Neuron-specific variables dictionary
neuronvars = {
    'RS': {
        'desc': 'Cortical regular-spiking neuron',
        'vars_US': [charge, potential, deflection, gas, iNa_gates, iK_gate, iM_gate],
        'vars_elec': [charge, potential, iNa_gates, iK_gate, iM_gate],
        'Vm0': -71.9  # mV
    },
    'FS': {
        'desc': 'Cortical fast-spiking neuron',
        'vars_US': [charge, potential, deflection, gas, iNa_gates, iK_gate, iM_gate],
        'vars_elec': [charge, potential, iNa_gates, iK_gate, iM_gate],
        'Vm0': -71.4  # mV
    },
    'LTS': {
        'desc': 'Cortical, low-threshold spiking neuron',
        'vars_US': [charge, potential, deflection, gas, iNa_gates, iK_gate, iM_gate, iCa_gates],
        'vars_elec': [charge, potential, iNa_gates, iK_gate, iM_gate, iCa_gates],
        'Vm0': -54.0  # mV
    },
    'RE': {
        'desc': 'Thalamic reticular neuron',
        'vars_US': [charge, potential, deflection, gas, iNa_gates, iK_gate, iCa_gates],
        'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gates],
        'Vm0': -89.5  # mV
    },
    'TC': {
        'desc': 'Thalamo-cortical neuron',
        'vars_US': [charge, potential, deflection, gas, iNa_gates, iK_gate, iCa_gates,
                    iH_reg_factor, Ca_conc],
        'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gates, iH_reg_factor, Ca_conc],
        'Vm0': -61.93  # mV
    }
    # 'LeechT': {
    #     'desc': 'Leech "touch" neuron',
    #     'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gate, NaPump_reg, iKCa_reg],
    #     'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gate, NaPump_reg, iKCa_reg]
    # },
    # 'LeechP': {
    #     'desc': 'Leech "pressure" neuron',
    #     'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gate, iKCa2_gate,
    #                 NaPump2_reg, CaPump2_reg],
    #     'vars_elec': [charge, potential, iNa_gates, iK_gate, iCa_gate, iKCa2_gate, NaPump2_reg,
    #                   CaPump2_reg]
    # }
}
