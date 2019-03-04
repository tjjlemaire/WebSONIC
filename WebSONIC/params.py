# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 16:41:14

''' Definition of application parameters. '''

import numpy as np
import colorlover as cl

from .pltutils import PlotVariable, Current, CellType


# --------------------------------- Input parameters ---------------------------------

inputparams = dict(
    coverages=np.linspace(20.0, 100.0, 5),  # %
    diams=np.logspace(np.log10(16.0), np.log10(64.0), 3) * 1e-9,  # m
    US_freqs=np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6]),  # Hz
    US_amps=np.array([10, 20, 40, 60, 80, 100, 300, 600]) * 1e3,  # Pa
    elec_amps=np.array([-25, -10, -5, -2, 2, 5, 10, 25]),  # mA/m2
    PRFs=np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3]),  # Hz
    DCs=np.array([1., 5., 10., 25., 50., 75., 100.]),  # %
    tstim=250  # ms
)

inputdefaults = dict(
    coverages=100.,  # %
    diams=32.0e-9,  # m
    US_freqs=500e3,  # Hz
    US_amps=80e3,  # Pa
    elec_amps=10,  # mA/m2
    PRFs=1e2,  # Hz
    DCs=100.  # %
)

# --------------------------------- Plotting parameters ---------------------------------

pltparams = dict(
    tbounds=(-5., 300.),  # ms
    colorset=[item for i, item in enumerate(cl.scales['8']['qual']['Set1']) if i != 5]
)

ngraphs = 3


# --------------------------------- Current types ---------------------------------

iNa = Current('iNa', 'Depolarizing Sodium current', gates=['m', 'h'])
iKd = Current('iKd', 'Delayed-recifier Potassium current', gates=['n'])
iM = Current('iM', 'Slow non-inactivating Potassium current', gates=['p'])
iCaT = Current('iCaT', 'Low-threshold (T-type) Calcium current', gates=['s', 'u'])
iCaTs = Current('iCaTs', 'Low-threshold (Ts-type) Calcium current', gates=['s', 'u'])
iCaL = Current('iCaL', 'Long-lasting (L-type) Calcium current', gates=['q', 'r'])
iH = Current('iH', 'Hyperpolarization-activated mixed cationic current',
             gates=['O', 'OL = 1 - O - C'],
             internals=[
                 PlotVariable('P0', 'iH regulating factor (P0)'),
                 PlotVariable('C_Ca', 'Sumbmembrane Ca2+ concentration', label='[Ca2+]',
                              unit='uM', factor=1e6, bounds=(0, 150.0))
             ])
iKleak = Current('iKleak', 'Leakage Potassium current')
iLeak = Current('iLeak', 'Leakage current')


# --------------------------------- Cell types ---------------------------------

RS = CellType('RS', 'Cortical regular spiking neuron', [iNa, iKd, iM, iLeak], Vm0=-71.9)
FS = CellType('FS', 'Cortical fast spiking neuron', [iNa, iKd, iM, iLeak], Vm0=-71.4)
LTS = CellType('LTS', 'Cortical low-threshold spiking neuron', [iNa, iKd, iM, iCaT, iLeak], Vm0=-54.0)
IB = CellType('IB', 'Cortical intrinsically bursting neuron', [iNa, iKd, iM, iCaL, iLeak], Vm0=-71.4)
RE = CellType('RE', 'Thalamic reticular neuron', [iNa, iKd, iCaTs, iLeak], Vm0=-89.5)
TC = CellType('TC', 'Thalamo-cortical neuron', [iNa, iKd, iCaT, iH, iKleak, iLeak], Vm0=-61.93)

# LeechT = CellType('LeechT', 'Leech "touch" neuron', [???], Vm0=???)
# LeechP = CellType('LeechP', 'Leech "pressure" neuron', [???], Vm0=???)

# Neuron-specific variables dictionary
celltypes = {cell.name: cell for cell in [RS, FS, LTS, IB, RE, TC]}


# --------------------------------- Leech specific variables ---------------------------------

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
