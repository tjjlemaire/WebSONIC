# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-10 15:34:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-06 16:23:35

''' Definition of application parameters. '''

import numpy as np

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
    tstim=1000.0  # ms
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

# --------------------------------- Variables ---------------------------------

P0 = PlotVariable('P0', 'iH regulating factor (P0)', bounds=(-0.1, 1.1))
C_Ca = PlotVariable('C_Ca', 'Sub-membrane Ca2+ concentration', label='[Ca2+]', unit='uM',
                    factor=1e6, bounds=(0, 150.0))
C_Ca2 = PlotVariable('C_Ca', 'Sub-membrane Ca2+ concentration', label='[Ca2+]', unit='uM',
                     factor=1e6, bounds=(0, 0.4))

# --------------------------------- Current types ---------------------------------

# From Plaksin 2016
iNa = Current('iNa', 'Depolarizing Sodium current', gates=['m', 'h'])
iKd = Current('iKd', 'Delayed-recifier Potassium current', gates=['n'])
iM = Current('iM', 'Slow non-inactivating Potassium current', gates=['p'])
iCaT = Current('iCaT', 'Low-threshold (T-type) Calcium current', gates=['s', 'u'])
iCaL = Current('iCaL', 'Long-lasting (L-type) Calcium current', gates=['q', 'r'])
iH = Current('iH', 'Hyperpolarization-activated mixed cationic current', gates=['O', 'OL'],
             internals=[P0, C_Ca])
iKLeak = Current('iKLeak', 'Leakage Potassium current')
iLeak = Current('iLeak', 'Leakage current')

# From Tarnaud 2018
iA = Current('iA', 'A-type Potassium current', gates=['a', 'b'])
iCaT2 = Current('iCaT', 'Low-threshold (T-type) Calcium current', gates=['p', 'q'])
iCaL2 = Current('iCaL', 'Long-lasting (L-type) Calcium current', gates=['c', 'd1', 'd2'],
                internals=[C_Ca2])
iCaK = Current('iCaK', 'Calcium activated Potassium current', gates=['r'])


# --------------------------------- Cell types ---------------------------------

RS = CellType('RS', 'Cortical regular spiking', [iNa, iKd, iM, iLeak], Vm0=-71.9)
FS = CellType('FS', 'Cortical fast spiking', [iNa, iKd, iM, iLeak], Vm0=-71.4)
LTS = CellType('LTS', 'Cortical low-threshold spiking', [iNa, iKd, iM, iCaT, iLeak], Vm0=-54.0)
IB = CellType('IB', 'Cortical intrinsically bursting', [iNa, iKd, iM, iCaL, iLeak], Vm0=-71.4)
RE = CellType('RE', 'Thalamic reticular', [iNa, iKd, iCaT, iLeak], Vm0=-89.5)
TC = CellType('TC', 'Thalamo-cortical', [iNa, iKd, iCaT, iH, iKLeak, iLeak], Vm0=-61.93)
STN = CellType('STN', 'Sub-thalamic nucleus', [iNa, iKd, iA, iCaT2, iCaL2, iCaK, iLeak], Vm0=-58.0)

# Neuron-specific variables dictionary
celltypes = {cell.name: cell for cell in [STN, RS, FS, LTS, IB, RE, TC]}
